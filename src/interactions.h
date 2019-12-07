#pragma once

#include "intersections.h"

//////////
/* Random number generator */
// reference: https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A

// Generates a seed for a random number generator from 2 inputs plus a backoff
__host__ __device__
unsigned int initRand(unsigned int val0, unsigned int val1, unsigned int backoff = 16)
{
    unsigned int v0 = val0, v1 = val1, s0 = 0;

    for (unsigned int n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Takes our seed, updates it, and returns a pseudorandom float in [0..1]
__host__ __device__
float nextRand(unsigned int& s)
{
    s = (1664525u * s + 1013904223u);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}
//////////

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, unsigned int& seed) {
    float up = sqrt(nextRand(seed)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = nextRand(seed) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
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

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        unsigned int& seed) {

    pathSegment.ray.origin = intersect + 1e-4f * normal;    // New ray shoot from intersection point
    pathSegment.remainingBounces--;                         // Decrease bounce counter

    if (m.hasRefractive) {                                  // Refreaction
    float eta = 1.0f / m.indexOfRefraction;
    float unit_projection = glm::dot(pathSegment.ray.direction, normal);
    if (unit_projection > 0) {
        eta = 1.0f / eta;
    }

    // Schlick's approximation
    float R0 = powf((1.0f - eta) / (1.0f + eta), 2.0f);
    float R = R0 + (1 - R0) * powf(1 - glm::abs(unit_projection), 5.0f);
    if (R < nextRand(seed)) {
        // Refracting Light
        pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, eta);
        normal = -normal;
        pathSegment.color *= m.color;
    } else {
        // Reflecting Light
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.color;
    }
    } else if (nextRand(seed) < m.hasReflective) {                // Specular
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.color;
    } else {                                                // Diffusive
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, seed);
        pathSegment.diffuse = true;
    }
}
