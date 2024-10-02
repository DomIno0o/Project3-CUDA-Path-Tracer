#include "interactions.h"

__device__ extern glm::vec2 sampleRandomStratified(glm::vec2 uniform, int numSample);
__device__ extern glm::vec2 transformToDisk(const glm::vec2 squareInput);

__device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng, int iter)
{
    thrust::uniform_real_distribution<float> u01{ 0, 1 };

    glm::vec2 stratified{ sampleRandomStratified(glm::vec2{ u01(rng), u01(rng) }, iter) };

    float up = sqrt(stratified.x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = stratified.y * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng, int iter)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasRefractive)
    {
        float cosTheta = - glm::dot(pathSegment.ray.direction, normal);
        float sinTheta = sqrtf(1 - cosTheta * cosTheta);

        bool isEntering = cosTheta > 0;
        float iorRatio;
        if (isEntering) iorRatio = 1 / m.indexOfRefraction;
        else iorRatio = m.indexOfRefraction;

        // Total reflection
        if (iorRatio * sinTheta > 1)
        {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else
        {
            glm::vec3 dirOutPerp = iorRatio * (pathSegment.ray.direction + cosTheta * normal);
            glm::vec3 dirOutParallel = - sqrtf(1 - glm::dot(dirOutPerp, dirOutPerp)) * normal;
            pathSegment.ray.direction = dirOutPerp + dirOutParallel;
        }
    }
    else if (m.hasReflective > 0)
    {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else
    {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng, iter);
    }
    pathSegment.ray.origin = intersect + 0.0001f * pathSegment.ray.direction;
}
