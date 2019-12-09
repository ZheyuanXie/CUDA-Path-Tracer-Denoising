#pragma once
#include "sceneStructs.h"

struct BVHPrimitive {
	int index;
	BoundingBox bounds;
	glm::vec3 centroid;
	
	BVHPrimitive() {}
	BVHPrimitive(int ind, const BoundingBox &b) : 
		index(ind), bounds(b.minCorner, b.maxCorner), centroid((0.5f * (b.minCorner + b.maxCorner))) {}
};

//Surface Area Heuristic
struct SAH {
	int count = 0;
	BoundingBox bounds;
};

class BVHTreeNode {	
public:
	BoundingBox box;
	BVHTreeNode *leftchild, *rightchild;
	int Axis;
	int first_prime_off;	// first primetive in this array
	int primitive_count;	// the length of the array
	
	BVHTreeNode() {}

	void MakeLeaf(int first, int n, const BoundingBox &b) {
		leftchild = rightchild = nullptr;
		first_prime_off = first;
		primitive_count = n;
		box = b;
		Axis = -1;
	}
	
	void MakeNode(int axis, BVHTreeNode *l, BVHTreeNode *r) {
		leftchild = l;
		rightchild = r;
		box = (l->box || r->box);
		Axis = axis;
		primitive_count = 0;
	}
};

// Array of bvh node
struct BVH_ArrNode {
	BoundingBox bounds;
	int primitive_count;  
	int axis;          
	int primitivesOffset;   // leaf
	int rightchildoffset;  // interior
};

BVH_ArrNode* BuildBVHTree(int& totalNodes,std::vector<Triangle> &primitives);		   
void DeleteBVH(BVH_ArrNode *nodes);
