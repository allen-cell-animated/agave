SystemJS.config({
    map: {
        "imgui-js": "https://rawgit.com/flyover/imgui-js/master"
    },
    packages: {
        "imgui-js": {
            main: "imgui.js"
        }
    }
});
let ImGui;
let ImGui_Impl;
Promise.resolve().then(() => {
  return System.import("imgui-js").then((module) => {
    ImGui = module;
    return ImGui.default();
  });
}).then(() => {
  return System.import("imgui-js/example/imgui_impl").then((module) => {
    ImGui_Impl = module;
  });
}).then(() => {
        const canvas = document.getElementById("output");
        const devicePixelRatio = window.devicePixelRatio || 1;
        canvas.width = canvas.scrollWidth * devicePixelRatio;
        canvas.height = canvas.scrollHeight * devicePixelRatio;
        window.addEventListener("resize", () => {
            const devicePixelRatio = window.devicePixelRatio || 1;
            canvas.width = canvas.scrollWidth * devicePixelRatio;
            canvas.height = canvas.scrollHeight * devicePixelRatio;
        });

        ImGui.CreateContext();
        ImGui_Impl.Init(canvas);

        ImGui.StyleColorsDark();
        //ImGui.StyleColorsClassic();

        const clear_color = new ImGui.ImVec4(0.45, 0.55, 0.60, 1.00);

        /* static */
        let buf = "Quick brown fox";
        /* static */
        let f = 0.6;
    
        let done = false;
        window.requestAnimationFrame(_loop);

        function _loop(time) {
            ImGui_Impl.NewFrame(time);
            ImGui.NewFrame();

            ImGui.SetNextWindowPos(new ImGui.ImVec2(0, 0), ImGui.Cond.FirstUseEver);
            ImGui.SetNextWindowSize(new ImGui.ImVec2(294, 140), ImGui.Cond.FirstUseEver);
            ImGui.Begin("OME TIF Viewer");

            try {
                //ImGui.Text(`Hello, world ${123}`);
                ImGui.InputText("File", 
                    (_ = uiState.file) => {
                        if (_ !== uiState.file) {
                            uiState.file = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LOAD_OME_TIF", _);
                            flushCommandBuffer(cb);
                        }
                        return _;
                    }, 294
                );
                let resolutions = ["256x256", "512x512", "1024x1024", "1024x768"];
                ImGui.Combo("Resolution", 
                    (value = resolutions.indexOf(uiState.resolution)) => {
                        if (value !== uiState.resolution)
                        {
                            uiState.resolution = value;
                            var res = resolutions[value].match(/(\d+)x(\d+)/);
                            if (res.length === 3) {
                                res[0] = parseInt(res[1]);
                                res[1] = parseInt(res[2]);
                                var imgholder = document.getElementById("imageA");
                                imgholder.width = res[0];
                                imgholder.height = res[1];
                                imgholder.style.width = res[0];
                                imgholder.style.height = res[1];
                    
                                var cb = new commandBuffer();
                                cb.addCommand("SET_RESOLUTION", res[0], res[1]);
                                flushCommandBuffer(cb);
                            }
                        }
                        return value;
                    },
                    resolutions, 4
                );

                if (ImGui.Button("Reset Camera"))
                {
                    uiState.resetCamera();
                }
                ImGui.SliderFloat("Density", 
                    (_ = uiState.density) => {
                        if (_ !== uiState.density) {
                            uiState.density = _;
                            var cb = new commandBuffer();
                            cb.addCommand("DENSITY", _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 100.0
                );
                ImGui.ColorEdit4("clear color", clear_color);

                if (ImGui.CollapsingHeader("Camera")) {
                    ImGui.SliderFloat("exposure", 
                    (_ = uiState.exposure) => {
                        if (_ !== uiState.exposure) {
                            uiState.exposure = _;
                            var cb = new commandBuffer();
                            cb.addCommand("EXPOSURE", _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("aperture", 
                    (_ = uiState.aperture) => {
                        if (_ !== uiState.aperture) {
                            uiState.aperture = _;
                            var cb = new commandBuffer();
                            cb.addCommand("APERTURE", _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 0.1
                    );
                    ImGui.SliderFloat("focal_distance", 
                    (_ = uiState.focal_distance) => {
                        if (_ !== uiState.focal_distance) {
                            uiState.focal_distance = _;
                            var cb = new commandBuffer();
                            cb.addCommand("FOCALDIST", _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.1, 5.0
                    );
                    ImGui.SliderFloat("fov", 
                    (_ = uiState.fov) => {
                        if (_ !== uiState.fov) {
                            uiState.fov = _;
                            var cb = new commandBuffer();
                            cb.addCommand("FOV_Y", _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 90.0
                    );
                }

                if (ImGui.CollapsingHeader("Volume Clipping")) {
                    ImGui.SliderFloat("xmin", 
                    (_ = uiState.xmin) => {
                        if (_ !== uiState.xmin) {
                            uiState.xmin = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("xmax", 
                    (_ = uiState.xmax) => {
                        if (_ !== uiState.xmax) {
                            uiState.xmax = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("ymin", 
                    (_ = uiState.ymin) => {
                        if (_ !== uiState.ymin) {
                            uiState.ymin = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("ymax", 
                    (_ = uiState.ymax) => {
                        if (_ !== uiState.ymax) {
                            uiState.ymax = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("zmin", 
                    (_ = uiState.zmin) => {
                        if (_ !== uiState.zmin) {
                            uiState.zmin = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );
                    ImGui.SliderFloat("zmax", 
                    (_ = uiState.zmax) => {
                        if (_ !== uiState.zmax) {
                            uiState.zmax = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SET_CLIP_REGION", uiState.xmin, uiState.xmax, uiState.ymin, uiState.ymax, uiState.zmin, uiState.zmax);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 1.0
                    );

                }

                if (ImGui.CollapsingHeader("Lighting")) {

                    if (ImGui.ColorEdit3("skyTopColor", uiState.skyTopColor)) {
                        var cb = new commandBuffer();
                        cb.addCommand("SKYLIGHT_TOP_COLOR", 
                            uiState["skyTopIntensity"] * uiState.skyTopColor[0],
                            uiState["skyTopIntensity"] * uiState.skyTopColor[1],
                            uiState["skyTopIntensity"] * uiState.skyTopColor[2]);
                        flushCommandBuffer(cb);                
                    }

                    ImGui.SliderFloat("skyTopIntensity", 
                    (_ = uiState.skyTopIntensity) => {
                        if (_ !== uiState.skyTopIntensity) {
                            uiState.skyTopIntensity = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SKYLIGHT_TOP_COLOR",
                                uiState["skyTopColor"][0] * _,
                                uiState["skyTopColor"][1] * _,
                                uiState["skyTopColor"][2] * _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.01, 100.0
                    );

                    if (ImGui.ColorEdit3("skyMidColor", uiState.skyMidColor)) {
                        var cb = new commandBuffer();
                        cb.addCommand("SKYLIGHT_MIDDLE_COLOR", 
                            uiState["skyMidIntensity"] * uiState.skyMidColor[0],
                            uiState["skyMidIntensity"] * uiState.skyMidColor[1],
                            uiState["skyMidIntensity"] * uiState.skyMidColor[2]);
                        flushCommandBuffer(cb);                
                    }

                    ImGui.SliderFloat("skyMidIntensity", 
                    (_ = uiState.skyMidIntensity) => {
                        if (_ !== uiState.skyMidIntensity) {
                            uiState.skyMidIntensity = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SKYLIGHT_MIDDLE_COLOR",
                                uiState["skyMidColor"][0] * _,
                                uiState["skyMidColor"][1] * _,
                                uiState["skyMidColor"][2] * _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.01, 100.0
                    );

                    if (ImGui.ColorEdit3("skyBotColor", uiState.skyBotColor)) {
                        var cb = new commandBuffer();
                        cb.addCommand("SKYLIGHT_BOTTOM_COLOR", 
                            uiState["skyBotIntensity"] * uiState.skyBotColor[0],
                            uiState["skyBotIntensity"] * uiState.skyBotColor[1],
                            uiState["skyBotIntensity"] * uiState.skyBotColor[2]);
                        flushCommandBuffer(cb);                
                    }

                    ImGui.SliderFloat("skyBotIntensity", 
                    (_ = uiState.skyBotIntensity) => {
                        if (_ !== uiState.skyBotIntensity) {
                            uiState.skyBotIntensity = _;
                            var cb = new commandBuffer();
                            cb.addCommand("SKYLIGHT_BOTTOM_COLOR",
                                uiState["skyBotColor"][0] * _,
                                uiState["skyBotColor"][1] * _,
                                uiState["skyBotColor"][2] * _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.01, 100.0
                    );

                    ImGui.SliderFloat("lightDistance", 
                    (_ = uiState.lightDistance) => {
                        if (_ !== uiState.lightDistance) {
                            uiState.lightDistance = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LIGHT_POS", 0, _, uiState["lightTheta"] * 180.0 / 3.14159265, uiState["lightPhi"] * 180.0 / 3.14159265);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 100.0
                    );
                    ImGui.SliderFloat("lightTheta", 
                    (_ = uiState.lightTheta) => {
                        if (_ !== uiState.lightTheta) {
                            uiState.lightTheta = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LIGHT_POS", 0, uiState["lightDistance"], _ * 180.0 / 3.14159265, uiState["lightPhi"] * 180.0 / 3.14159265);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    -180.0, 180.0
                    );
                    ImGui.SliderFloat("lightPhi", 
                    (_ = uiState.lightPhi) => {
                        if (_ !== uiState.lightPhi) {
                            uiState.lightPhi = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LIGHT_POS", 0, uiState["lightDistance"], uiState["lightTheta"] * 180.0 / 3.14159265, _ * 180.0 / 3.14159265);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.0, 180.0
                    );

                    ImGui.SliderFloat("lightSize", 
                    (_ = uiState.lightSize) => {
                        if (_ !== uiState.lightSize) {
                            uiState.lightSize = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LIGHT_SIZE", 0, _, _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.01, 100.0
                    );

                    ImGui.SliderFloat("lightIntensity", 
                    (_ = uiState.lightIntensity) => {
                        if (_ !== uiState.lightIntensity) {
                            uiState.lightIntensity = _;
                            var cb = new commandBuffer();
                            cb.addCommand("LIGHT_COLOR", 0, uiState["lightColor"][0] * _, uiState["lightColor"][1] * _, uiState["lightColor"][2] * _);
                            flushCommandBuffer(cb);                
                            //_stream_mode_suspended = true;
                        }
                        return _;
                    },
                    0.01, 100.0
                    );

                    if (ImGui.ColorEdit3("lightColor", uiState.lightColor)) {
                        var cb = new commandBuffer();
                        cb.addCommand("LIGHT_COLOR", 0, 
                            uiState.lightColor[0] * uiState["lightIntensity"], 
                            uiState.lightColor[1] * uiState["lightIntensity"], 
                            uiState.lightColor[2] * uiState["lightIntensity"]
                        );
                        flushCommandBuffer(cb);                
                    }

                }

            } catch (e) {
                ImGui.TextColored(new ImGui.ImVec4(1.0, 0.0, 0.0, 1.0), "error: ");
                ImGui.SameLine();
                ImGui.Text(e.message);
            }

            ImGui.End();

            ImGui.SetNextWindowPos(new ImGui.ImVec2(294, 0), ImGui.Cond.FirstUseEver);
            ImGui.SetNextWindowSize(new ImGui.ImVec2(294, 140), ImGui.Cond.FirstUseEver);
            ImGui.Begin("Image Channels");
            ImGui.End();


            ImGui.EndFrame();

            ImGui.Render();

            const gl = ImGui_Impl.gl;
            gl && gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl && gl.clearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            gl && gl.clear(gl.COLOR_BUFFER_BIT);
            //gl.useProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound

            ImGui_Impl.RenderDrawData(ImGui.GetDrawData());

            window.requestAnimationFrame(done ? _done : _loop);
        }

        function _done() {
            ImGui_Impl.Shutdown();
            ImGui.DestroyContext();
        }
    });