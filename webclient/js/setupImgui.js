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
Promise.all([
        System.import("imgui-js"),
        System.import("imgui-js/example/imgui_impl")
    ])
    .then((modules) => {
        const ImGui = modules[0];
        const ImGui_Impl = modules[1];

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
            } catch (e) {
                ImGui.TextColored(new ImGui.ImVec4(1.0, 0.0, 0.0, 1.0), "error: ");
                ImGui.SameLine();
                ImGui.Text(e.message);
            }

            ImGui.End();

            ImGui_Impl.EndFrame();

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