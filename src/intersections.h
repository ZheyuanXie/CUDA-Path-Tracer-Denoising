#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define BOUNDING_VOLUME_INTERSECTION_CULLING

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/**
 * Test intersection between a ray and a transformed mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
// __host__ __device__ float meshIntersectionTest(Geom mesh, Triangle* triangles, Ray r,
//   glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

//   glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
//   glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

//   Ray rt;
//   rt.origin = ro;
//   rt.direction = rd;

// #ifdef BOUNDING_VOLUME_INTERSECTION_CULLING
//   // Test ray AABB intersection
//   // Reference: https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
//   // r.dir is unit direction vector of ray
//   glm::vec3 dirfrac;
//   dirfrac.x = 1.0f / rt.direction.x;
//   dirfrac.y = 1.0f / rt.direction.y;
//   dirfrac.z = 1.0f / rt.direction.z;
//   // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
//   // r.org is origin of ray
//   float t1 = (mesh.min_bound.x - rt.origin.x)*dirfrac.x;
//   float t2 = (mesh.max_bound.x - rt.origin.x)*dirfrac.x;
//   float t3 = (mesh.min_bound.y - rt.origin.y)*dirfrac.y;
//   float t4 = (mesh.max_bound.y - rt.origin.y)*dirfrac.y;
//   float t5 = (mesh.min_bound.z - rt.origin.z)*dirfrac.z;
//   float t6 = (mesh.max_bound.z - rt.origin.z)*dirfrac.z;

//   float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
//   float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

//   // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
//   if (tmax < 0 || tmin > tmax) {
//     return -1; 
//   }
// #endif // BOUNDING_VOLUME_INTERSECTION_CULLING

//   // find intersecting triangle with minimum t
//   float min_t = FLT_MAX;
//   int min_idx = -1;
//   for (int i = 0; i < mesh.num_triangles; i++) {
//     Triangle tri = triangles[i];
//     float t = -1;
//     glm::vec3 bary;
//     if (glm::intersectRayTriangle(rt.origin, rt.direction, tri.v[0], tri.v[1], tri.v[2], bary)) {
//       t = bary.z;
//       if (t > 0.0f && t < min_t) {
//         min_t = t;
//         min_idx = i;
//       }
//     };
//   }
  
//   // no intersection detected
//   if (min_idx == -1) {
//     return -1;
//   }

//   glm::vec3 objspaceIntersection = getPointOnRay(rt, min_t);
//   intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
//   normal = glm::normalize(multiplyMV(mesh.transform, glm::vec4(triangles[min_idx].n, 0.f)));
//   outside = (glm::dot(rt.origin, normal) < 0);
//   return min_t;
// }

// traverse all the mesh to get intersection!
__host__ __device__ float meshIntersectionTest(Geom mesh, Triangle* tris,
	Ray r, glm::vec2 &uv, glm::vec3 &normal,
	glm::mat3 &tangToWorld, bool &outside) {
	int min_ind = -1;	// the nearest triangle id
	float tmin = FLT_MAX;
	glm::vec3 bari(0.f), minbari(0.f);

	//traverse all the triangles
	for (int i = mesh.T_startidx; i < mesh.T_endidx; i++) {
		//the baryPosition output uses barycentric coordinates for the x and y components.The z component is the scalar factor for ray.
		//That is, 1.0 - baryPosition.x - baryPosition.y = actual z barycentric coordinate
		if (glm::intersectRayTriangle(r.origin, r.direction, tris[i].verts[0].pos, tris[i].verts[1].pos, tris[i].verts[2].pos, bari)) {
			if (bari.z > 0.f && bari.z < tmin) {
				min_ind = i;
				tmin = bari.z;
				minbari = bari;
			}
		}
	}
	//not hit anything
	if (min_ind == -1) { return -1; }

	normal = (1.0f - minbari.x - minbari.y) * tris[min_ind].verts[0].normal +
		minbari.x * tris[min_ind].verts[1].normal +
		minbari.y * tris[min_ind].verts[2].normal;
	normal = glm::normalize(normal);

	uv = (1.0f - minbari.x - minbari.y) * tris[min_ind].verts[0].uv +
		minbari.x * tris[min_ind].verts[1].uv +
		minbari.y * tris[min_ind].verts[2].uv;

	//TODO: for calculating the normal
	glm::vec3 interp = getPointOnRay(r, tmin);
	glm::vec3 deltaPos1 = tris[min_ind].verts[1].pos - interp;
	glm::vec3 deltaPos2 = tris[min_ind].verts[2].pos - interp;
	glm::vec2 deltaUV1 = tris[min_ind].verts[1].uv - uv;
	glm::vec2 deltaUV2 = tris[min_ind].verts[2].uv - uv;
	glm::vec3 tangent = glm::normalize((deltaPos1*deltaUV2.y - deltaPos2 * deltaUV1.y) * (1.f / (deltaUV1.x*deltaUV2.y - deltaUV1.y*deltaUV2.x)));
	glm::vec3 bitangent = glm::normalize((deltaPos2*deltaUV1.x - deltaPos1 * deltaUV2.x) * (1.f / (deltaUV1.x*deltaUV2.y - deltaUV1.y*deltaUV2.x)));
	tangToWorld = glm::mat3(tangent, bitangent, normal);

	return tmin;
}

#define MAX_BVH_DEPTH 64
__host__ __device__ bool IntersectBVH(const Ray &ray, ShadeableIntersection * isect,
	int & hit_tri_index,
	const BVH_ArrNode *bvh_nodes,
	const Triangle* primitives) {

	if (bvh_nodes == nullptr) { return false; }

	bool hit = false;
	int isDirNeg[3] = { ray.direction.x < 0.f, ray.direction.y < 0.f, ray.direction.z < 0.f };
	glm::vec3 invdir(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

	int toVisitOffset = 0, curr_ind = 0;
	int needToVisit[MAX_BVH_DEPTH];
	while (true) {
		const BVH_ArrNode *node = &bvh_nodes[curr_ind];
		float temp_t = 0.f;
		//check bounding box intersection
		if (node->bounds.AABBIntersect2(ray, invdir)) {
			if (node->primitive_count > 0) {												// leaf node
				for (int i = 0; i < node->primitive_count; i++) {							// intersect test with each triangle in the nodes
					ShadeableIntersection inter;
					if (primitives[node->primitivesOffset + i].Intersect(ray, &inter)) {	// triangles intersect test
						hit = true;
						if (isect->t == -1.0f) {
							(*isect) = inter;
							hit_tri_index = primitives[node->primitivesOffset + i].id;
						}
						else {
							if (inter.t < isect->t) {
								(*isect) = inter;
								hit_tri_index = primitives[node->primitivesOffset + i].id;
							}
						}
					}
				}
				if (toVisitOffset == 0) { break; }
				curr_ind = needToVisit[--toVisitOffset];
			}
			else {
				// Trick: learn from hanming zhang, if toVisitOffset reaches maximum
				// we don't want add more index to needToVisit Array
				// we just give up this interior node and handle previous nodes instead 
				if (toVisitOffset == MAX_BVH_DEPTH) {
					curr_ind = needToVisit[--toVisitOffset];
					continue;
				}
				// add index to nodes to visit
				if (isDirNeg[node->axis]) {
					needToVisit[toVisitOffset++] = curr_ind + 1;
					curr_ind = node->rightchildoffset;
				}
				else {
					needToVisit[toVisitOffset++] = node->rightchildoffset;
					curr_ind = curr_ind + 1;
				}
			}
		}
		else {// do not hit anything
			if (toVisitOffset == 0) { break; }
			curr_ind = needToVisit[--toVisitOffset];
		}
	}
	return hit;
}
