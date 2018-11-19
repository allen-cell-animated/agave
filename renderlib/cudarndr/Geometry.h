#pragma once

#include "Spectrum.h"

#include "helper_math.cuh"

#include <algorithm>
#include <math.h>
#include <stdio.h>

class CColorRgbHdr;
class Vec2i;
class Vec2f;
class Vec3i;
class Vec4i;
class Vec4f;
class CColorXyz;

class CColorRgbHdr
{
public:
	HOD CColorRgbHdr(void)
	{
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
	}

	HOD CColorRgbHdr(const float& r, const float& g, const float& b)
	{
		this->r = r;
		this->g = g;
		this->b = b;
	}

	HOD CColorRgbHdr(const float& rgb)
	{
		r = rgb;
		g = rgb;
		b = rgb;
	}

	HOD CColorRgbHdr& operator = (const CColorRgbHdr& p)			
	{
		r = p.r;
		g = p.g;
		b = p.b;

		return *this;
	}

	HOD CColorRgbHdr& operator = (const CColorXyz& S);	

	HOD CColorRgbHdr& operator += (CColorRgbHdr &p)		
	{
		r += p.r;
		g += p.g;
		b += p.b;	

		return *this;
	}

	HOD CColorRgbHdr operator * (float f) const
	{
		return CColorRgbHdr(r * f, g * f, b * f);
	}

	HOD CColorRgbHdr& operator *= (float f)
	{
		for (int i = 0; i < 3; i++)
			(&r)[i] *= f;

		return *this;
	}

	HOD CColorRgbHdr operator / (float f) const
	{
		float inv = 1.0f / f;
		return CColorRgbHdr(r * inv, g * inv, b * inv);
	}

	HOD CColorRgbHdr& operator /= (float f)
	{
		float inv = 1.f / f;
		r *= inv; g *= inv; b *= inv;
		return *this;
	}

	HOD float operator[](int i) const
	{
		return (&r)[i];
	}

	HOD float operator[](int i)
	{
		return (&r)[i];
	}

	HOD bool Black(void)
	{
		return r == 0.0f && g == 0.0f && b == 0.0f;
	}

	HOD CColorRgbHdr Pow(float e)
	{
		return CColorRgbHdr(powf(r, e), powf(g, e), powf(b, e));
	}

	HOD void FromXYZ(float x, float y, float z)
	{
		const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
		const float gWeight[3] = {-0.969256f,  1.875991f,  0.041556f };
		const float bWeight[3] = { 0.055648f, -0.204043f,  1.057311f };

		r =	rWeight[0] * x +
			rWeight[1] * y +
			rWeight[2] * z;

		g =	gWeight[0] * x +
			gWeight[1] * y +
			gWeight[2] * z;

		b =	bWeight[0] * x +
			bWeight[1] * y +
			bWeight[2] * z;
	}

	HOD CColorXyz ToXYZ(void) const
	{
		return CColorXyz::FromRGB(r, g, b);
	}

	HOD CColorXyza ToXYZA(void) const
	{
		return CColorXyza::FromRGB(r, g, b);
	}

	void PrintSelf(void)
	{
		printf("[%g, %g, %g]\n", r, g, b);
	}

	float	r;
	float	g;
	float	b;
};


class Vec2f
{
public:
	HOD Vec2f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
	}

	HOD Vec2f(const float& x, const float& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2f(const float& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	HOD Vec2f(const Vec2f& v)
	{
		this->x = v.x;
		this->y = v.y;
	}

	HOD float operator[](int i) const
	{
		return (&x)[i];
	}

	HOD float& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec2f& operator = (const Vec2f& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	HOD Vec2f operator + (const Vec2f& v) const
	{
		return Vec2f(x + v.x, y + v.y);
	}

	HOD Vec2f& operator += (const Vec2f& v)
	{
		x += v.x; y += v.y;
		return *this;
	}

	HOD Vec2f operator - (const Vec2f& v) const
	{
		return Vec2f(x - v.x, y - v.y);
	}

	HOD Vec2f& operator -= (const Vec2f& v)
	{
		x -= v.x; y -= v.y;
		return *this;
	}

	HOD Vec2f operator * (float f) const
	{
		return Vec2f(x * f, y * f);
	}

	HOD Vec2f& operator *= (float f)
	{
		x *= f; 
		y *= f; 

		return *this;
	}

	HOD bool operator < (const Vec2f& V) const
	{
		return V.x < x && V.y < y;
	}

	HOD bool operator > (const Vec2f& V) const
	{
		return V.x > x && V.y > y;
	}

	HOD bool operator == (const Vec2f& V) const
	{
		return V.x == x && V.y == y;
	}

	HOD float LengthSquared(void) const
	{
		return x * x + y * y;
	}

	HOD float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	void PrintSelf(void)
	{
		printf("[%g, %g]\n", x, y);
	}

	float x, y;
};

class Vec2i
{
public:
	HOD Vec2i(void)
	{
		this->x = 0;
		this->y = 0;
	}

	HOD Vec2i(const int& x, const int& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2i(int& x, int& y)
	{
		this->x = x;
		this->y = y;
	}

	HOD Vec2i(const Vec2f& V)
	{
		this->x = (int)V.x;
		this->y = (int)V.y;
	}

	HOD Vec2i(const int& xy)
	{
		this->x = xy;
		this->y = xy;
	}

	HOD int operator[](int i) const
	{
		return (&x)[i];
	}

	HOD int& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec2i& operator = (const Vec2i& v)
	{
		x = v.x; 
		y = v.y; 

		return *this;
	}

	HOD bool operator < (const Vec2i& V) const
	{
		return V.x < x && V.y < y;
	}

	HOD bool operator > (const Vec2i& V) const
	{
		return V.x > x && V.y > y;
	}

	HOD bool operator == (const Vec2i& V) const
	{
		return V.x == x && V.y == y;
	}

	void PrintSelf(void)
	{
		printf("[%d, %d]\n", x, y);
	}

	int x, y;
};

class Vec3i
{
public:
	HOD Vec3i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	HOD Vec3i(const int& x, const int& y, const int& z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	HOD Vec3i(const int& xyz)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
	}

	HOD int operator[](int i) const
	{
		return (&x)[i];
	}

	HOD int& operator[](int i)
	{
		return (&x)[i];
	}

	HOD Vec3i& operator = (const Vec3i &v)
	{
		x = v.x; 
		y = v.y; 
		z = v.z;

		return *this;
	}

	HOD bool operator < (const Vec3i& V) const
	{
		return V.x < x && V.y < y && V.z < z;
	}

	HOD bool operator > (const Vec3i& V) const
	{
		return V.x > x && V.y > y && V.z > z;
	}

	HOD bool operator == (const Vec3i& V) const
	{
		return V.x == x && V.y == y && V.z == z;
	}

	HOD float LengthSquared(void) const
	{
		return (float)(x * x + y * y);
	}

	HOD float Length(void) const
	{
		return sqrtf(LengthSquared());
	}

	HOD int Max(void)
	{
		if (x >= y && x >= z)
		{
			return x;
		}		
		else
		{
			if (y >= x && y >= z)
				return y;
			else
				return z;
		}
	}

	HOD int Min(void)
	{
		if (x <= y && x <= z)
		{
			return x;
		}		
		else
		{
			if (y <= x && y <= z)
				return y;
			else
				return z;
		}
	}

	void PrintSelf(void)
	{
		printf("[%d, %d, %d]\n", x, y, z);
	}

	int x, y, z;
};


class Vec4i
{
public:
	HOD Vec4i(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
		this->w = 0;
	}

	HOD Vec4i(const int& x, const int& y, const int& z, const int& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	HOD Vec4i(const int& xyzw)
	{
		this->x = xyzw;
		this->y = xyzw;
		this->z = xyzw;
		this->w = xyzw;
	}

	int x, y, z, w;
};

class Vec4f
{
public:
	HOD Vec4f(void)
	{
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
		this->w = 0.0f;
	}

	HOD Vec4f(const float& x, const float& y, const float& z, const float& w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	HOD Vec4f operator * (float f) const
	{
		return Vec4f(x * f, y * f, z * f, w * f);
	}

	HOD Vec4f& operator *= (float f)
	{
		x *= f; 
		y *= f; 
		z *= f;
		w *= f;

		return *this;
	}

	void PrintSelf(void)
	{
		printf("[%f, %f, %f, %f]\n", x, y, z, w);
	}

	float x, y, z, w;
};

static HOD CColorRgbHdr operator * (const CColorRgbHdr& v, const float& f) 			{ return CColorRgbHdr(f * v.r, f * v.g, f * v.b); 					};
static HOD CColorRgbHdr operator * (const float& f, const CColorRgbHdr& v) 			{ return CColorRgbHdr(f * v.r, f * v.g, f * v.b); 					};
static HOD CColorRgbHdr operator * (const CColorRgbHdr& p1, const CColorRgbHdr& p2) 	{ return CColorRgbHdr(p1.r * p2.r, p1.g * p2.g, p1.b * p2.b); 		};
static HOD CColorRgbHdr operator + (const CColorRgbHdr& a, const CColorRgbHdr& b)		{ return CColorRgbHdr(a.r + b.r, a.g + b.g, a.b + b.b);				};

// Vec2f
static inline HOD Vec2f operator * (const Vec2f& v, const float& f) 	{ return Vec2f(f * v.x, f * v.y);					};
static inline HOD Vec2f operator * (const float& f, const Vec2f& v) 	{ return Vec2f(f * v.x, f * v.y);					};
static inline HOD Vec2f operator * (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x * v2.x, v1.y * v2.y);			};
static inline HOD Vec2f operator / (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x / v2.x, v1.y / v2.y);			};
// static inline HOD Vec2f operator - (const Vec2f& v1, const Vec2f& v2) 	{ return Vec2f(v1.x - v2.x, v1.y - v2.y);			};

static inline HOD Vec2f operator * (Vec2f& V2f, Vec2i& V2i)	{ return Vec2f(V2f.x * V2i.x, V2f.y * V2i.y);				};

class CRay
{	
public:
	// ToDo: Add description
	HOD CRay(void)
	{
	};

	// ToDo: Add description
	HOD CRay(float3 Origin, float3 Dir, float MinT, float MaxT = INF_MAX, int PixelID = 0)
	{
        m_O = Origin;
        m_D = Dir;
        m_MinT		= MinT;
		m_MaxT		= MaxT;
		m_PixelID	= PixelID;
	}

	// ToDo: Add description
	HOD ~CRay(void)
	{
	}

	// ToDo: Add description
	HOD CRay& operator=(const CRay& Other)
	{
		m_O			= Other.m_O;
		m_D			= Other.m_D;
		m_MinT		= Other.m_MinT;
		m_MaxT		= Other.m_MaxT;
		m_PixelID	= Other.m_PixelID;

		// By convention, always return *this
		return *this;
	}

	// ToDo: Add description
	HOD float3 operator()(float t) const
	{
		return m_O + normalize(m_D) * t;
	}

	void PrintSelf(void)
	{
		//printf("Origin ");
		//m_O.PrintSelf();

		//printf("Direction ");
		//m_D.PrintSelf();

		printf("Min T: %4.2f\n", m_MinT);
		printf("Max T: %4.2f\n", m_MaxT);
		printf("Pixel ID: %d\n", m_PixelID);
	}

	float3 	m_O;			/*!< Ray origin */
	float3 	m_D;			/*!< Ray direction */
	float	m_MinT;			/*!< Minimum parametric range */
	float	m_MaxT;			/*!< Maximum parametric range */
	int		m_PixelID;		/*!< Pixel ID associated with the ray */
};

class CSize2D
{
public:
	Vec2f	m_Size;
	Vec2f	m_InvSize;

	HOD CSize2D(void) :
		m_Size(1.0f, 1.0f),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	HOD CSize2D(const float& X, const float& Y) :
		m_Size(X, Y),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	HOD CSize2D(const Vec2f& V) :
		m_Size(V),
		m_InvSize(1.0f / m_Size.x, 1.0f / m_Size.y)
	{
	};

	// ToDo: Add description
	HOD CSize2D& operator=(const CSize2D& Other)
	{
		m_Size		= Other.m_Size;
		m_InvSize	= Other.m_InvSize;

		return *this;
	}

	HOD void Update(void)
	{
		m_InvSize = Vec2f(1.0f / m_Size.x, 1.0f / m_Size.y);
	}
};


class CRange
{
public:
	HOD CRange(void) :
		m_Min(0.0f),
		m_Max(100.0f),
		m_Range(m_Max - m_Min),
		m_InvRange(1.0f / m_Range)
	{
	}

	HOD CRange(const float& MinRange, const float& MaxRange) :
		m_Min(MinRange),
		m_Max(MaxRange),
		m_Range(m_Max - m_Min),
		m_InvRange(1.0f / m_Range)
	{
	}

	HOD CRange& operator = (const CRange& Other)
	{
		m_Min		= Other.m_Min; 
		m_Max		= Other.m_Max; 
		m_Range		= Other.m_Range;
		m_InvRange	= Other.m_InvRange;

		return *this;
	}

	HOD float	GetMin(void) const			{ return m_Min;								}
	HOD void	SetMin(const float& Min)	{ m_Min = Min; m_Range = m_Max - m_Min;		}
	HOD float	GetMax(void) const			{ return m_Max;								}
	HOD void	SetMax(const float& Max)	{ m_Max = Max; m_Range = m_Max - m_Min;		}
	HOD float	GetRange(void) const		{ return m_Range;							}
	HOD float	GetInvRange(void) const		{ return m_InvRange;						}

	void PrintSelf(void)
	{
		printf("[%4.2f - %4.2f]\n", m_Min, m_Max);
	}

private:
	float	m_Min;
	float	m_Max;
	float	m_Range;
	float	m_InvRange;
};

class CPixel
{
public:
	HOD CPixel(void)
	{
		m_XY	= Vec2i(256);
		m_ID	= 0;
	}

	HOD CPixel(const Vec2f& ImageXY, const Vec2i& Resolution)
	{
		m_XY	= Vec2i((int)floorf(ImageXY.x), (int)floorf(ImageXY.y));
		m_ID	= (m_XY.y * Resolution.x) + m_XY.x;
	}

	HOD CPixel& operator = (const CPixel& Other)
	{
		m_XY	= Other.m_XY; 
		m_ID	= Other.m_ID;

		return *this;
	}

	Vec2i	m_XY;		/*!< Pixel coordinates */
	int		m_ID;		/*!< Pixel ID */
};


HOD inline CColorRgbHdr& CColorRgbHdr::operator = (const CColorXyz& S)			
{
	r = S.c[0];
	g = S.c[1];
	b = S.c[2];

	return *this;
}

HOD inline float AbsDot(const float3& a, const float3& b)
{
    return fabsf(dot(a, b));
};


// reflect
inline HOD float Fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline HOD float Fminf(float a, float b)
{
	return a < b ? a : b;
}

inline HOD float Clamp(float f, float a, float b)
{
	return Fmaxf(a, Fminf(f, b));
}

HOD inline float Length(float3 p1)
{
    return sqrt(dot(p1, p1));
}
HOD inline float LengthSquared(float3 p1)
{
    return dot(p1, p1);
}
HOD inline float DistanceSquared(float3 p1, float3 p2)
{
    return LengthSquared(p1 - p2);
}

HOD inline void CreateCS(const float3& N, float3& u, float3& v)
{
	if ((N.x == 0) && (N.y == 0))
	{
		if (N.z < 0.0f)
			u = make_float3(-1.0f, 0.0f, 0.0f);
		else
			u = make_float3(1.0f, 0.0f, 0.0f);
		
		v = make_float3(0.0f, 1.0f, 0.0f);
	}
	else
	{
		// Note: The root cannot become zero if
		// N.x == 0 && N.y == 0.
		const float d = 1.0f / sqrtf(N.y*N.y + N.x*N.x);
		
		u = make_float3(N.y * d, -N.x * d, 0);
		v = cross(N, u);
	}
}


HOD inline CColorRgbHdr Lerp(float T, const CColorRgbHdr& C1, const CColorRgbHdr& C2)
{
	const float OneMinusT = 1.0f - T;
	return CColorRgbHdr(OneMinusT * C1.r + T * C2.r, OneMinusT * C1.g + T * C2.g, OneMinusT * C1.b + T * C2.b);
}

HOD inline CColorXyz Lerp(float T, const CColorXyz& C1, const CColorXyz& C2)
{
	const float OneMinusT = 1.0f - T;
	return CColorXyz(OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0]);
}

// ToDo: Add description
class CTransferFunction
{
public:
	float			m_P[MAX_NO_TF_POINTS];		/*!< Node positions */
	CColorRgbHdr	m_C[MAX_NO_TF_POINTS];		/*!< Node colors in HDR RGB */
	int				m_NoNodes;					/*!< No. nodes */

	// ToDo: Add description
	HO CTransferFunction(void)
	{
		for (int i = 0; i < MAX_NO_TF_POINTS; i++)
		{
			m_P[i]	= 0.0f;
			m_C[i]	= SPEC_BLACK;
		}

		m_NoNodes = 0;
	}

	// ToDo: Add description
	HO ~CTransferFunction(void)
	{
	}

	// ToDo: Add description
	HOD CTransferFunction& operator=(const CTransferFunction& Other)
	{
		for (int i = 0; i < MAX_NO_TF_POINTS; i++)
		{
			m_P[i]	= Other.m_P[i];
			m_C[i]	= Other.m_C[i];
		}

		m_NoNodes = Other.m_NoNodes;

		return *this;
	}

	// ToDo: Add description
	HOD CColorRgbHdr F(const float& P)
	{
		for (int i = 0; i < m_NoNodes - 1; i++)
		{
			if (P >= m_P[i] && P < m_P[i + 1])
			{
				const float T = (float)(P - m_P[i]) / (m_P[i + 1] - m_P[i]);
				return Lerp(T, m_C[i], m_C[i + 1]);
			}
		}

		return CColorRgbHdr(0.0f);
	}
};

// ToDo: Add description
class CTransferFunctions
{
public:
	CTransferFunction	m_Opacity;
	CTransferFunction	m_Diffuse;
	CTransferFunction	m_Specular;
	CTransferFunction	m_Emission;
	CTransferFunction	m_Roughness;

	// ToDo: Add description
	HO CTransferFunctions(void)
	{
	}

	// ToDo: Add description
	HO ~CTransferFunctions(void)
	{
	}

	// ToDo: Add description
	HOD CTransferFunctions& operator=(const CTransferFunctions& Other)
	{
		m_Opacity		= Other.m_Opacity;
		m_Diffuse		= Other.m_Diffuse;
		m_Specular		= Other.m_Specular;
		m_Emission		= Other.m_Emission;
		m_Roughness		= Other.m_Roughness;

		return *this;
	}
};

inline float RandomFloat(void)
{
	return (float)rand() / RAND_MAX;
}

template <class T>
class CVec2
{
public:

	T	m_D[3];
};

template <class T>
class CVec3
{
public:

	T	m_D[3];
};

template <class T>
class CVec4
{
public:

	T	m_D[3];
};
