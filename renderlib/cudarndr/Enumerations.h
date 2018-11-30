#pragma once

// View modes
enum EViewMode
{
	ViewModeUndefined = 0,
	
	ViewModeUser,

	// Normal views
	ViewModeFront,
	ViewModeBack,
	ViewModeLeft,
	ViewModeRight,
	ViewModeTop,
	ViewModeBottom,

	// Isometric views
	ViewModeIsometricFrontLeftTop,
	ViewModeIsometricFrontRightTop,
	ViewModeIsometricFrontLeftBottom,
	ViewModeIsometricFrontRightBottom,
	ViewModeIsometricBackLeftTop,
	ViewModeIsometricBackRightTop,
	ViewModeIsometricBackLeftBottom,
	ViewModeIsometricBackRightBottom
};

enum EDirty
{
	MaterialsDirty			= 0x00001,
	TexturesDirty			= 0x00002,
	CameraDirty				= 0x00004,
	GeometryDirty			= 0x00008,
	AccelerationDirty		= 0x00010,
	BitmapsDirty			= 0x00020,
	VolumeDirty				= 0x00040,
	FrameBufferDirty		= 0x00080,
	RenderParamsDirty		= 0x00100,
	VolumeDataDirty			= 0x00200,
	FilmResolutionDirty		= 0x00400,
	EnvironmentDirty		= 0x00800,
	FocusDirty				= 0x01000,
	LightsDirty				= 0x02000,
	BenchmarkDirty			= 0x04000,
	TransferFunctionDirty	= 0x08000,
	AnimationDirty			= 0x10000,
	RoiDirty				= 0x20000,
	MeshDirty				= 0x40000,
};

enum EContainment
{
	ContainmentNone,
	ContainmentPartial,
	ContainmentFull
};

enum EAxis
{
	AxisX = 0,
	AxisY,
	AxisZ,
	AxisUndefined
};