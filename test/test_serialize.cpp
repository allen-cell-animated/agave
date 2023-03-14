#include <catch2/catch_test_macros.hpp>

#include "../agave_app/Serialize.h"
#include "renderlib/json/json.hpp"

// #include <array>
// #include <vector>

TEST_CASE("Json Serialization", "[serialize]")
{
  SECTION("Read and write PathTraceSettings_V1")
  {
    auto settings = PathTraceSettings_V1{};
    settings.primaryStepSize = 0.1f;
    settings.secondaryStepSize = 0.2f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<PathTraceSettings_V1>();
    REQUIRE(settings == settings2);
  }
  SECTION("Read and write TimelineSettings_V1")
  {
    auto settings = TimelineSettings_V1{};
    settings.minTime = 0.1f;
    settings.maxTime = 0.2f;
    settings.currentTime = 0.3f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<TimelineSettings_V1>();
    REQUIRE(settings == settings2);
  }
  SECTION("Read and write CameraSettings_V1")
  {
    auto settings = CameraSettings_V1{};
    settings.eye = { 0.1f, 0.2f, 0.3f };
    settings.target = { 0.4f, 0.5f, 0.6f };
    settings.up = { 0.7f, 0.8f, 0.9f };
    settings.projection = Projection::PERSPECTIVE;
    settings.fovY = 0.1f;
    settings.orthoScale = 0.2f;
    settings.exposure = 0.3f;
    settings.aperture = 0.4f;
    settings.focalDistance = 0.5f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<CameraSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ControlPointSettings_V1")
  {
    auto settings = ControlPointSettings_V1{};
    settings.x = 0.1f;
    settings.value = { 0.2f, 0.3f, 0.4f, 0.5f };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ControlPointSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write LutParams_V1")
  {
    auto settings = LutParams_V1{};
    settings.window = 0.1f;
    settings.level = 0.2f;
    settings.isovalue = 0.3f;
    settings.isorange = 0.4f;
    settings.pctLow = 0.5f;
    settings.pctHigh = 0.6f;
    settings.controlPoints = { { 0.1f, { 0.2f, 0.3f, 0.4f, 0.5f } }, { 0.2f, { 0.3f, 0.4f, 0.5f, 0.6f } } };
    settings.mode = GradientEditMode::CUSTOM;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<LutParams_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ChannelSettings_V1")
  {
    auto settings = ChannelSettings_V1{};
    settings.enabled = true;
    settings.lutParams = { 0.1f,
                           0.2f,
                           0.3f,
                           0.4f,
                           0.5f,
                           0.6f,
                           { { 0.1f, { 0.2f, 0.3f, 0.4f, 0.5f } }, { 0.2f, { 0.3f, 0.4f, 0.5f, 0.6f } } },
                           GradientEditMode::CUSTOM };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ChannelSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write LightSettings_V1")
  {
    auto settings = LightSettings_V1{};
    settings.type = LightType::AREA;
    settings.distance = 0.1;
    settings.theta = 0.2;
    settings.phi = 0.3;
    settings.color = { 0.1f, 0.2f, 0.3f };
    settings.colorIntensity = 0.4f;
    settings.topColor = { 0.5f, 0.6f, 0.7f };
    settings.topColorIntensity = 0.8f;
    settings.middleColor = { 0.9f, 0.1f, 0.2f };
    settings.middleColorIntensity = 0.3f;
    settings.bottomColor = { 0.9f, 0.1f, 0.2f };
    settings.bottomColorIntensity = 0.3f;
    settings.width = 0.4f;
    settings.height = 0.5f;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<LightSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write CaptureSettings_V1")
  {
    auto settings = CaptureSettings_V1{};
    settings.width = 1;
    settings.height = 2;
    settings.filenamePrefix = "test";
    settings.outputDirectory = "test2";
    settings.samples = 3;
    settings.seconds = 4.0;
    settings.durationType = DurationType::TIME;
    settings.startTime = 6;
    settings.endTime = 7;
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<CaptureSettings_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Read and write ViewerState_V1")
  {
    auto settings = ViewerState_V1{};
    settings.name = "test";
    settings.version = { 1, 2, 3 };
    settings.resolution = { 1, 2 };
    settings.renderIterations = 3;
    settings.pathTracer = PathTraceSettings_V1{
      0.1f,
      0.2f,
    };
    settings.timeline = TimelineSettings_V1{ 0.1, 0.2, 0.3 };
    settings.scene = 1;
    settings.clipRegion = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    settings.scale = { 0.1f, 0.2f, 0.3f };
    settings.camera = CameraSettings_V1{};
    settings.backgroundColor = { 0.1f, 0.2f, 0.3f };
    settings.showBoundingBox = true;
    settings.channels = { ChannelSettings_V1{}, ChannelSettings_V1{} };
    settings.density = 100.0;
    settings.lights = { LightSettings_V1{}, LightSettings_V1{} };
    nlohmann::json json = settings;
    std::string str = json.dump();
    json = nlohmann::json::parse(str);
    auto settings2 = json.get<ViewerState_V1>();
    REQUIRE(settings == settings2);
  }

  SECTION("Error out on bad json")
  {
    // valid json to struct fails
    auto jstr = "{}";
    auto json = nlohmann::json::parse(jstr);
    REQUIRE_THROWS_AS(json.get<PathTraceSettings_V1>(), std::exception);

    // all good
    jstr = R"(
      {
        "primaryStepSize": 0.1,
        "secondaryStepSize": 0.2
      }
    )";
    json = nlohmann::json::parse(jstr);
    REQUIRE(json.get<PathTraceSettings_V1>() == PathTraceSettings_V1{ 0.1, 0.2 });

    // json parse fails
    jstr = R"(
      {
        "primaryStepSize": 0.1,
        "secondaryStepSize": 0.2,
      }
    )";
    REQUIRE_THROWS_AS(nlohmann::json::parse(jstr), std::exception);
  }

  SECTION("Read v1 json with extra fields")
  {
    // "capture" is an extra field here and will be dropped
    auto jstr = R"(
{
    "backgroundColor": [
        0,
        0,
        0
    ],
    "boundingBoxColor": [
        1,
        1,
        1
    ],
    "camera": {
        "aperture": 0,
        "exposure": 0.75,
        "eye": [
            0.7656243443489075,
            -0.4507044553756714,
            1.197649359703064
        ],
        "focalDistance": 0.75,
        "fovY": 55,
        "orthoScale": 0.5,
        "projection": 0,
        "target": [
            0.5,
            0.4440000057220459,
            0.21950767934322357
        ],
        "up": [
            0.11175604164600372,
            0.7481830716133118,
            0.6540123820304871
        ]
    },
    "capture": {
        "durationType": 1,
        "endTime": 0,
        "filenamePrefix": "frame",
        "height": 0,
        "outputDirectory": "/Users/danielt/Documents",
        "samples": 32,
        "seconds": 10,
        "startTime": 0,
        "width": 0
    },
    "channels": [
        {
            "diffuseColor": [
                1,
                0,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                1,
                1,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                1,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                0.14589150249958038,
                0.5000076293945312
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.29179826378822327,
                0.5000076293945312,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0,
                0.43768978118896484
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                0.5000076293945312,
                0.4164034426212311
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0.2705119550228119,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        }
    ],
    "clipRegion": [
        [
            0,
            1
        ],
        [
            0,
            1
        ],
        [
            0,
            1
        ]
    ],
    "density": 8.5,
    "lights": [
        {
            "bottomColor": [
                1,
                1,
                1
            ],
            "bottomColorIntensity": 1,
            "color": [
                10,
                10,
                10
            ],
            "colorIntensity": 1,
            "distance": 1,
            "height": 1,
            "middleColor": [
                1,
                1,
                1
            ],
            "middleColorIntensity": 1,
            "phi": 1.5707963705062866,
            "theta": 0,
            "topColor": [
                1,
                1,
                1
            ],
            "topColorIntensity": 1,
            "type": 1,
            "width": 1
        },
        {
            "bottomColor": [
                10,
                10,
                10
            ],
            "bottomColorIntensity": 1,
            "color": [
                1,
                1,
                1
            ],
            "colorIntensity": 100,
            "distance": 10,
            "height": 1,
            "middleColor": [
                10,
                10,
                10
            ],
            "middleColorIntensity": 1,
            "phi": 1.5707963705062866,
            "theta": 0,
            "topColor": [
                10,
                10,
                10
            ],
            "topColorIntensity": 1,
            "type": 0,
            "width": 1
        }
    ],
    "name": "/Users/danielt/Downloads/files-d99bba176cdc737434ef1774bfdc3255/AICS-12_1070_71092.ome.tif",
    "pathTracer": {
        "primaryStepSize": 4,
        "secondaryStepSize": 4
    },
    "renderIterations": 2086,
    "resolution": [
        619,
        622
    ],
    "scale": [
        0.10833299905061722,
        0.10833299905061722,
        0.28999999165534973
    ],
    "scene": 0,
    "showBoundingBox": true,
    "timeline": {
        "currentTime": 0,
        "maxTime": 0,
        "minTime": 0
    },
    "version": [
        1,
        4,
        1
    ]
}

    )";

    auto json = nlohmann::json::parse(jstr);
    auto settings = json.get<ViewerState_V1>();
  }

  SECTION("Read valid 1.4.1 json")
  {
    std::string jstr = R"(
{
    "backgroundColor": [
        0,
        0,
        0
    ],
    "boundingBoxColor": [
        1,
        1,
        1
    ],
    "camera": {
        "aperture": 0,
        "exposure": 0.75,
        "eye": [
            0.06596243381500244,
            -0.508243203163147,
            1.0754612684249878
        ],
        "focalDistance": 0.75,
        "fovY": 55,
        "orthoScale": 0.5,
        "projection": 0,
        "target": [
            0.5,
            0.4440000057220459,
            0.21950767934322357
        ],
        "up": [
            0.2748546004295349,
            0.5704407095909119,
            0.7739846706390381
        ]
    },
    "channels": [
        {
            "diffuseColor": [
                1,
                0,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                1,
                1,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                1,
                1
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": true,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                0.14589150249958038,
                0.5000076293945312
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.29179826378822327,
                0.5000076293945312,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0,
                0.43768978118896484
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0,
                0.5000076293945312,
                0.4164034426212311
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        },
        {
            "diffuseColor": [
                0.5000076293945312,
                0.2705119550228119,
                0
            ],
            "emissiveColor": [
                0,
                0,
                0
            ],
            "enabled": false,
            "glossiness": 1,
            "lutParams": {
                "controlPoints": [
                    {
                        "value": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "x": 0
                    },
                    {
                        "value": [
                            1,
                            1,
                            1,
                            1
                        ],
                        "x": 1
                    }
                ],
                "isorange": 0.10000000149011612,
                "isovalue": 0.5,
                "level": 0.5,
                "mode": 2,
                "pctHigh": 0.9800000190734863,
                "pctLow": 0.5,
                "window": 0.25
            },
            "opacity": 1,
            "specularColor": [
                0,
                0,
                0
            ]
        }
    ],
    "clipRegion": [
        [
            0,
            1
        ],
        [
            0,
            1
        ],
        [
            0,
            1
        ]
    ],
    "density": 8.5,
    "lights": [
        {
            "bottomColor": [
                1,
                1,
                1
            ],
            "bottomColorIntensity": 1,
            "color": [
                10,
                10,
                10
            ],
            "colorIntensity": 1,
            "distance": 1,
            "height": 1,
            "middleColor": [
                1,
                1,
                1
            ],
            "middleColorIntensity": 1,
            "phi": 1.5707963705062866,
            "theta": 0,
            "topColor": [
                1,
                1,
                1
            ],
            "topColorIntensity": 1,
            "type": 1,
            "width": 1
        },
        {
            "bottomColor": [
                10,
                10,
                10
            ],
            "bottomColorIntensity": 1,
            "color": [
                1,
                1,
                1
            ],
            "colorIntensity": 100,
            "distance": 10,
            "height": 1,
            "middleColor": [
                10,
                10,
                10
            ],
            "middleColorIntensity": 1,
            "phi": 1.5707963705062866,
            "theta": 0,
            "topColor": [
                10,
                10,
                10
            ],
            "topColorIntensity": 1,
            "type": 0,
            "width": 1
        }
    ],
    "name": "/Users/danielt/Downloads/files-d99bba176cdc737434ef1774bfdc3255/AICS-12_1070_71092.ome.tif",
    "pathTracer": {
        "primaryStepSize": 4,
        "secondaryStepSize": 4
    },
    "renderIterations": 1031,
    "resolution": [
        619,
        622
    ],
    "scale": [
        0.10833299905061722,
        0.10833299905061722,
        0.28999999165534973
    ],
    "scene": 0,
    "showBoundingBox": true,
    "timeline": {
        "currentTime": 0,
        "maxTime": 0,
        "minTime": 0
    },
    "version": [
        1,
        4,
        1
    ]
}

      )";
    auto json = nlohmann::json::parse(jstr);
    auto settings = json.get<ViewerState_V1>();
  }
}
