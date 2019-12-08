#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
#define PiOver2			  1.57079632679489661923f
#define PiOver4			  0.78539816339744830961f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f

#define InvPi			  0.31830988618379067154;
#define Inv2Pi			  0.15915494309189533577;
#define Inv4Pi			  0.07957747154594766788;

typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;
typedef glm::vec3 Normal3f;
typedef glm::vec3 Direction3f;
typedef glm::vec2 UV2f;

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
	extern void compareThreeVertex(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3, glm::vec3 &min, glm::vec3 &max);
}
