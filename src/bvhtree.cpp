#include "bvhtree.h"
#include <algorithm>
#include <iostream>

int MaxPrimsInNode = 10;
int cmp_axis = 0;
bool comp(const BVHPrimitive &a, const BVHPrimitive &b) {
	return a.centroid[cmp_axis] < b.centroid[cmp_axis];
}

//TODO: not work...
//int re = 0;
//int split_id = 0;
//BoundingBox temp(glm::vec3(0.f), glm::vec3(0.f));
//bool partition_method(const BVHPrimitive &p) {
//	int b = re * temp.getOffset(p.centroid)[cmp_axis];
//	if (b == re) { b = re - 1; }
//	return b <= split_id;
//}

BVHTreeNode* Build(std::vector<BVHPrimitive> &primitive,
	std::vector<Triangle> &orderedT,
	std::vector<Triangle> &tris,
	int start, int end, int& totalNodes)  {
	BVHTreeNode *node = new BVHTreeNode();
	totalNodes++;

	BoundingBox bounds = primitive[start].bounds;				// Compute bounds of all tris in BVH node
	for (int i = start; i < end; i++) {
		bounds = bounds || primitive[i].bounds;
	}

	int ntris = end - start;
	if (ntris == 1) {											// is a LEAF
		int first_prime_off = orderedT.size();
		for (int i = start; i < end; i++) {
			orderedT.push_back(tris[primitive[i].index]);
		}
		node->MakeLeaf(first_prime_off, ntris, bounds);
		return node;
	}

	BoundingBox centro_b = BoundingBox(primitive[start].centroid, primitive[start].centroid);
	for (int i = start; i < end; i++) {
		centro_b = centro_b || primitive[i].centroid;
	}

	int axi = centro_b.CalculatethelongestAxis();
	
	// if the length axis is zero, then make these triangles in one node
	if (centro_b.maxCorner[axi] == centro_b.minCorner[axi]) { 
		int first_prime_off = orderedT.size();	
		for (int i = start; i < end; i++) {
			orderedT.push_back(tris[primitive[i].index]);
		}
		node->MakeLeaf(first_prime_off, ntris, bounds);
		return node;
	}

	// Surface Area Heuristic
	float mid;
	if (ntris == 2) { //just divide
		mid = (1.f * (start + end)) / 2.f;
		cmp_axis = axi;
		//https://www.geeksforgeeks.org/stdnth_element-in-cpp/
		std::nth_element(&primitive[start], &primitive[(int)mid], &primitive[end - 1] + 1, comp);
		node->MakeNode(axi, Build(primitive, orderedT, tris, start, (int)mid, totalNodes), Build(primitive, orderedT, tris, (int)mid, end, totalNodes));
	} 
	else
	{ 
		const int region_count = 9;
		SAH region[region_count];

		for (int i = start; i < end; i++) {
			int b = region_count * centro_b.getOffset(primitive[i].centroid)[axi];
			if (b == region_count) { b = region_count - 1; }
			region[b].bounds = region[b].bounds || primitive[i].bounds;
			region[b].count++;
		}

		// Compute costs for splitting after each region choose
		float cost[region_count - 1]; // we have region_count - 1 choose
		for (int i = 0; i < region_count - 1; i++) {
			int count_a = 0, count_b = 0;
			BoundingBox A, B;
			for (int j = 0; j <= i; j++) {			// 0 - i
				A = A || region[j].bounds;
				count_a += region[j].count;
			}
			for (int j = i + 1; j < region_count; j++) { // i - end
				B = B || region[j].bounds;
				count_b += region[j].count;
			}
			//represented by the area of bounding box
			cost[i] = 1.f + (count_a * A.Area() + count_b * B.Area()) / bounds.Area();
		}

		// Find the min cost for spliting
		float min_cost = FLT_MAX;
		int split_ind = 0;
		for (int i = 0; i < region_count - 1; i++) {
			if (cost[i] < min_cost) {
				min_cost = cost[i];
				split_ind = i;
			}
		}

		if (min_cost < ntris || ntris > MaxPrimsInNode) { // assume that origin method is ntris
			BVHPrimitive *midp = std::partition(&primitive[start], &primitive[end - 1] + 1, [=](const BVHPrimitive &pi) {
				int b = region_count * centro_b.getOffset(pi.centroid)[axi];
				if (b == region_count) { b = region_count - 1; }
				return b <= split_ind;
			});
			mid = midp - &primitive[0];
		} else {										 // still use origin method
			int first_prime_off = orderedT.size();
			for (int i = start; i < end; i++) {
				orderedT.push_back(tris[primitive[i].index]);
			}
			node->MakeLeaf(first_prime_off, ntris, bounds);
			return node;
		}
		node->MakeNode(axi, Build(primitive, orderedT, tris, start, (int)mid, totalNodes), Build(primitive, orderedT, tris, (int)mid, end, totalNodes));
	}
	return node;
}

int DFSBVHTree(BVHTreeNode *node, BVH_ArrNode *bvh_nodes, int &offset) {
	BVH_ArrNode *Node = &bvh_nodes[offset];
	Node->bounds = node->box;
	int new_off = offset++;

	if (node->primitive_count > 0) { // leaf
		Node->primitivesOffset = node->first_prime_off;
		Node->primitive_count = node->primitive_count;
		return new_off;
	} 

	Node->primitive_count = 0;
	Node->axis = node->Axis;
	DFSBVHTree(node->leftchild, bvh_nodes, offset);
	Node->rightchildoffset = DFSBVHTree(node->rightchild, bvh_nodes, offset);

	return new_off;
}

void DeleteBVHNode(BVHTreeNode *root) {
	if (root->leftchild == nullptr && root->rightchild == nullptr) {	// LEAF CONDITION
		delete root;
		return;
	}
	DeleteBVHNode(root->leftchild);		// left children
	DeleteBVHNode(root->rightchild);	// right children
	delete root;
	return;
}

BVH_ArrNode* BuildBVHTree(int& totalNodes, std::vector<Triangle> &tris) {
	if (tris.size() == 0) { return nullptr; }

	// Build BVH from tris
	std::vector<BVHPrimitive> primitives(tris.size());	  // GET THE INFO ARRAY
	for (int i = 0; i < tris.size(); i++) {			      // GIVE THE BOUNDINGBOX INFO
		primitives[i] = BVHPrimitive(i, tris[i].GetWorldBoundbox());
	}
	//BUILD BVH KDTREE
	std::vector<Triangle> orderedTris;
	orderedTris.reserve(tris.size());
	totalNodes = 0;

	BVHTreeNode *root;
	root = Build(primitives, orderedTris, tris, 0, tris.size(), totalNodes);
	tris.swap(orderedTris); // GET THIS TRIANGLE

	//DFS of the tree to get a linear code
	BVH_ArrNode *bvh_nodes = new BVH_ArrNode[totalNodes];
	int offset = 0;
	DFSBVHTree(root, bvh_nodes, offset);

	DeleteBVHNode(root);
	return bvh_nodes;
}


void DeleteBVH(BVH_ArrNode *nodes) {
	delete[] nodes;
}




