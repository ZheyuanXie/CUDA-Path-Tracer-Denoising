#include "boundingbox.h"

BoundingBox::BoundingBox() :
	minCorner(glm::vec3(0.f)), maxCorner(glm::vec3(0.f)) {
}

BoundingBox::BoundingBox(const glm::vec3& min, const glm::vec3& max):
	minCorner(min), maxCorner(max) {
}

BoundingBox::BoundingBox(const BoundingBox &b):
	minCorner(b.minCorner), maxCorner(b.maxCorner) {
}

float BoundingBox::Area() const {
	glm::vec3 d = maxCorner - minCorner;
	return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

bool BoundingBox::isPointInsideAABB(const glm::vec3 &vert) const {
	if ((vert.x >= minCorner.x && vert.x <= maxCorner.x) && 
		(vert.y >= minCorner.y && vert.y <= maxCorner.y) && 
		(vert.z >= minCorner.z && vert.z <= maxCorner.z)) {
		return true;
	} else {
		return false;
	}
}

int BoundingBox::CalculatethelongestAxis() {
	glm::vec3 distance = maxCorner - minCorner;
	if (distance.x > distance.y && distance.x > distance.z) {   // X
		return 0;
	} else if (distance.y > distance.z) {						// Y
		return 1;
	} else {													// Z
		return 2;
	}
}

glm::vec3 BoundingBox::getOffset(const glm::vec3 &p) const {
	glm::vec3 offset = p - minCorner;
	if (maxCorner.x > minCorner.x) {
		offset.x /= (maxCorner.x - minCorner.x);
	}
	if (maxCorner.y > minCorner.y) {
		offset.y /= (maxCorner.y - minCorner.y);
	}
	if (maxCorner.z > minCorner.z) {
		offset.z /= (maxCorner.z - minCorner.z);
	}
	return offset;
}
