#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

#define ERRORCHECK 1

#define ENABLE_ANTI_ALIASING
#define ENABLE_STREAMCOMPACTION
//#define ENABLE_MATERIAL_SORTING
//#define ENABLE_DEPTH_OF_FIELD
//#define ENABLE_STRATIFIED

#define NUM_CELLS_STRATIFIED 200

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__device__ glm::vec2 sampleRandomStratified(glm::vec2 uniform, int numSample)
{
#   ifdef ENABLE_STRATIFIED
        // Samples (int) sqrt(numTotalSamples) from a stratified distribution
        // If more samples are sampled, they will be random over the entire domain
        int numCellsPerSide = (int)sqrtf(NUM_CELLS_STRATIFIED);
        float gridLength = 1.0f / numCellsPerSide;

        if (numSample >= numCellsPerSide * numCellsPerSide) return uniform;

        glm::vec2 gridIdx;
        gridIdx.y = numSample / numCellsPerSide;
        gridIdx.x = numSample - gridIdx.y * numCellsPerSide;

        return (gridIdx + uniform) * gridLength;
#   else
        return uniform;
#   endif
}

__device__ glm::vec2 transformToDisk(const glm::vec2 squareInput)
{
    // taken from https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#sec:unit-disk-sample
    glm::vec2 offsetInput = 2.0f * squareInput - glm::vec2{ 1.0f };
    if (offsetInput.x == 0 && offsetInput.y == 0) return { 0, 0 };

    float r, theta;

    if (fabs(offsetInput.x) > fabs(offsetInput.y))
    {
        r = offsetInput.x;
        theta = PI_OVER_FOUR * offsetInput.x / offsetInput.y;
    }
    else
    {
        r = offsetInput.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * offsetInput.x / offsetInput.y;
    }

    return r * glm::vec2{ cosf(theta), sinf(theta) };
}

__host__ __device__ inline float convertLinearToSRGB(float linear)
{
    // taken from https://en.wikipedia.org/wiki/SRGB#Transformation
    if (linear <= 0.0031308f)
        return linear * 12.92f;
    else
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        // Clamp color to valid RGB values and apply gamma correction
        glm::ivec3 color;
        color.x = glm::clamp((int)(convertLinearToSRGB(pix.x / iter) * 255.0), 0, 255);
        color.y = glm::clamp((int)(convertLinearToSRGB(pix.y / iter) * 255.0), 0, 255);
        color.z = glm::clamp((int)(convertLinearToSRGB(pix.z / iter) * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static bool* dev_conditionBuffer = NULL;
static int* dev_keyBufferPaths = NULL;
static int* dev_keyBufferIntersections = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#   ifdef ENABLE_STREAMCOMPACTION
        cudaMalloc(&dev_conditionBuffer, pixelcount * sizeof(bool));
#   endif
#   ifdef ENABLE_MATERIAL_SORTING
        cudaMalloc(&dev_keyBufferPaths, pixelcount * sizeof(int));
        cudaMalloc(&dev_keyBufferIntersections, pixelcount * sizeof(int));
#   endif

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

#   ifdef ENABLE_STREAMCOMPACTION
        cudaFree(dev_conditionBuffer);
#   endif
#   ifdef ENABLE_MATERIAL_SORTING
        cudaFree(dev_keyBufferPaths);
        cudaFree(dev_keyBufferIntersections);
#   endif

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) return;

    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];
    Ray& ray = segment.ray;

    ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
#   ifdef ENABLE_ANTI_ALIASING
        thrust::default_random_engine rng_aa = makeSeededRandomEngine(iter, index, -1);
        thrust::uniform_real_distribution<float> aaOffset(0, 1);
        glm::vec2 aaOffsetVec = sampleRandomStratified(glm::vec2{ aaOffset(rng_aa), aaOffset(rng_aa) }, iter);
        ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f - aaOffsetVec.x)
            - cam.up    * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f - aaOffsetVec.y)
        );
#   else
        ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)(cam.resolution.x + 1) * 0.5f)
            - cam.up    * cam.pixelLength.y * ((float)y - (float)(cam.resolution.y + 1) * 0.5f)
        );
#   endif

#   ifdef ENABLE_DEPTH_OF_FIELD
        // use different random engines for DoF and AA
        thrust::default_random_engine rng_dof = makeSeededRandomEngine(iter, index, -2);
        thrust::uniform_real_distribution<float> dofUniform(0, 1);

        glm::vec2 aperturePoint = cam.aperture * transformToDisk(
            sampleRandomStratified(glm::vec2{ dofUniform(rng_dof), dofUniform(rng_dof) }, iter)
        );
        float perpendicularRayDirection = glm::dot(ray.direction, cam.view);
        float t = cam.focalDistance / perpendicularRayDirection;

        glm::vec3 focusPoint = ray.origin + t * ray.direction;
        ray.origin += aperturePoint.x * cam.right + aperturePoint.y * cam.up;
        ray.direction = glm::normalize(focusPoint - ray.origin);
#   endif

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths) return;

    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int numPaths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    /*static int bounceCount{0};
    static int stratifiedIdx[NUM_CELLS_STRATIFIED] = ;

    if (bounceCount >= NUM_CELLS_STRATIFIED)
    {

    }*/

    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) // if the intersection exists...
    {
        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 1.0f)
        {
            path.color *= (materialColor * material.emittance);
            // Assume that emittors do not reflect any light
            path.remainingBounces = 0;
        }
        // Otherwise, do some pseudo-lighting computation. This is actually more
        // like what you would expect from shading in a rasterizer like OpenGL.
        // TODO: replace this! you should be able to start with basically a one-liner
        else
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
            // thrust::uniform_real_distribution<float> u01(0, 1);

            // Compute intersection point on surface
            glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);

            if (!material.hasRefractive)
                path.color *= materialColor; // glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction)* materialColor;

            scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng, iter);

            --path.remainingBounces;
        }
    }
    else
    {
        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int pixelCount, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= pixelCount) return;

    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color;
}

__global__ void computeConditionBufferAndPartialImage(PathSegment* paths, int N, bool* conditionBuffer, glm::vec3* image)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    PathSegment iterationPath = paths[index];

    if (iterationPath.remainingBounces <= 0)
    {
        conditionBuffer[index] = true;
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
    else
    {
        conditionBuffer[index] = false;
    }
}

__global__ void computeKeyBuffers(const ShadeableIntersection* intersections, int N, int* keyBuffer1, int* keyBuffer2)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    int materialId = intersections[index].materialId;

    keyBuffer1[index] = materialId;
    keyBuffer2[index] = materialId;
}

dim3 computeBlockCount1D(unsigned int N, unsigned int blockSize)
{
    return dim3{ (N + blockSize - 1) / blockSize };
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelCount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth{ 0 };
    int numPaths{ pixelCount };
    PathSegment* dev_path_end{ dev_paths + numPaths };
    
    thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
    thrust::device_ptr<PathSegment> thrust_paths_end(dev_path_end);
    thrust::device_ptr<bool> thrust_conditionBuffer(dev_conditionBuffer);
    thrust::device_ptr<ShadeableIntersection> thrust_intersections(dev_intersections);
    thrust::device_ptr<int> thrust_keyBufferPaths(dev_keyBufferPaths);
    thrust::device_ptr<int> thrust_keyBufferIntersections(dev_keyBufferIntersections);

    dim3 numBlocks1d{ computeBlockCount1D(numPaths, blockSize1d) };

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    bool iterationComplete{ false };
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, numPaths * sizeof(ShadeableIntersection));

        // tracing
        computeIntersections<<<numBlocks1d, blockSize1d>>>(
            depth,
            numPaths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#       ifdef ENABLE_MATERIAL_SORTING
            computeKeyBuffers<<<numBlocks1d, blockSize1d>>>(
                dev_intersections, numPaths, dev_keyBufferPaths, dev_keyBufferIntersections);
            checkCUDAError("computeKey");

            // Sorting twice is faster than using zip_iterator and uses less memory
            // than sorting an index map and using gather
            thrust::sort_by_key(
                thrust_keyBufferPaths, thrust_keyBufferPaths + numPaths, thrust_paths);
            checkCUDAError("sorting paths");

            thrust::sort_by_key(
                thrust_keyBufferIntersections, thrust_keyBufferIntersections + numPaths, thrust_intersections);
            checkCUDAError("sorting intersections");
#       endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numBlocks1d, blockSize1d>>>(
            iter,
            numPaths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth
        );
        checkCUDAError("shadeFakeMaterial");
        cudaDeviceSynchronize();

#       ifdef ENABLE_STREAMCOMPACTION
            computeConditionBufferAndPartialImage<<<numBlocks1d, blockSize1d>>>(
                dev_paths, numPaths, dev_conditionBuffer, dev_image);

            // Removes entries with 1s in the conditionBuffer
            thrust_paths_end = thrust::remove_if(
                thrust_paths, thrust_paths + numPaths, thrust_conditionBuffer, thrust::identity<bool>());        
            cudaDeviceSynchronize();

            numPaths = thrust_paths_end - thrust_paths;
            numBlocks1d = computeBlockCount1D(numPaths, blockSize1d);

            // All rays have been terminated
            if (numPaths == 0) iterationComplete = true;
#       endif

        // Maximum depth reached
        if (depth == traceDepth) iterationComplete = true;

        if (guiData != NULL) guiData->TracedDepth = depth;
    }

#   ifdef ENABLE_STREAMCOMPACTION
        if (numPaths)
        {
            // Assemble the rest of this iteration and apply it to the image
            finalGather<<<numBlocks1d, blockSize1d>>>(numPaths, dev_image, dev_paths);
            checkCUDAError("finalGather");
        }

#   else
        // Assemble this iteration and apply it to the image
        finalGather<<<numBlocks1d, blockSize1d>>>(pixelCount, dev_image, dev_paths);
        checkCUDAError("finalGather");
#   endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    checkCUDAError("sendImageToPBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}