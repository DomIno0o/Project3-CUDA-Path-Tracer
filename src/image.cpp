#include <iostream>
#include <string>
#include <stb_image_write.h>

#include "image.h"

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new glm::vec3[x * y]) 
{}

Image::~Image()
{
    delete pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

inline float convertLinearToSRGB(float linear)
{
    // taken from https://en.wikipedia.org/wiki/SRGB#Transformation
    if (linear <= 0.0031308f)
        return linear * 12.92f;
    else
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
}

void Image::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1));
            bytes[3 * i + 0] = (unsigned char)(convertLinearToSRGB(pix.x) * 255.f);
            bytes[3 * i + 1] = (unsigned char)(convertLinearToSRGB(pix.y) * 255.f);
            bytes[3 * i + 2] = (unsigned char)(convertLinearToSRGB(pix.z) * 255.f);
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
