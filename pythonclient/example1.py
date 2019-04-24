import agaveclient

# imgplot = plt.imshow(numpy.zeros((1024, 768)))
if __name__ == '__main__':

    per_frame_commands = [
        ("LOAD_OME_TIF", "D:\\data\\april2019\\aligned_100s\\Interphase\\ACTB_36972_seg.ome.tif"),
        ("SET_RESOLUTION", 1024, 1024),
        ("SET_VOXEL_SCALE", 0.8, -0.8, 2.0),
        ("RENDER_ITERATIONS", 512),
        ("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1),
        ("EYE", 0.5, -0.5, 1.39614),
        ("TARGET", 0.5, -0.5, 0.0),
        ("UP", 0.0, 1.0, 0.0),
        ("FOV_Y", 55),
        ("EXPOSURE", 0.8714),
        ("DENSITY", 100),
        ("APERTURE", 0),
        ("FOCALDIST", 0.75),
        ("ENABLE_CHANNEL", 0, 1),
        ("MAT_DIFFUSE", 0, 1, 0, 1, 1.0),
        ("MAT_SPECULAR", 0, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 0, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 0, 0),
        ("SET_WINDOW_LEVEL", 0, 1, 0.758),
        ("ENABLE_CHANNEL", 1, 1),
        ("MAT_DIFFUSE", 1, 1, 1, 1, 1.0),
        ("MAT_SPECULAR", 1, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 1, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 1, 0),
        # ("SET_WINDOW_LEVEL", 1, 1, 0.7366),
        ("SET_WINDOW_LEVEL", 1, 1, 0.811),
        ("ENABLE_CHANNEL", 2, 1),
        ("MAT_DIFFUSE", 2, 0, 1, 1, 1.0),
        ("MAT_SPECULAR", 2, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 2, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 2, 0),
        ("SET_WINDOW_LEVEL", 2, 0.9922, 0.7704),
        ("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5),
        ("LIGHT_POS", 0, 10.1663, 1.1607, 0.5324),
        ("LIGHT_COLOR", 0, 122.926, 122.926, 125.999),
        ("LIGHT_SIZE", 0, 1, 1),
    ]
    per_frame_commands2 = [
        ("LOAD_OME_TIF", "D:\\data\\april2019\\aligned_100s\\Interphase\\TUBA1B_71126_raw.ome.tif"),
        ("SET_RESOLUTION", 1024, 1024),
        ("SET_VOXEL_SCALE", 0.8, -0.8, 2.0),
        ("RENDER_ITERATIONS", 512),
        ("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1),
        ("EYE", 0.5, -0.5, 1.39614),
        ("TARGET", 0.5, -0.5, 0.0),
        ("UP", 0.0, 1.0, 0.0),
        ("FOV_Y", 55),
        ("EXPOSURE", 0.8714),
        ("DENSITY", 100),
        ("APERTURE", 0),
        ("FOCALDIST", 0.75),
        ("ENABLE_CHANNEL", 0, 1),
        ("MAT_DIFFUSE", 0, 1, 0, 1, 1.0),
        ("MAT_SPECULAR", 0, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 0, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 0, 0),
        ("SET_WINDOW_LEVEL", 0, 1, 0.758),
        ("ENABLE_CHANNEL", 1, 1),
        ("MAT_DIFFUSE", 1, 1, 1, 1, 1.0),
        ("MAT_SPECULAR", 1, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 1, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 1, 0),
        # ("SET_WINDOW_LEVEL", 1, 1, 0.7366),
        ("SET_WINDOW_LEVEL", 1, 1, 0.811),
        ("ENABLE_CHANNEL", 2, 1),
        ("MAT_DIFFUSE", 2, 0, 1, 1, 1.0),
        ("MAT_SPECULAR", 2, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 2, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 2, 0),
        ("SET_WINDOW_LEVEL", 2, 0.9922, 0.7704),
        ("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5),
        ("LIGHT_POS", 0, 10.1663, 1.1607, 0.5324),
        ("LIGHT_COLOR", 0, 122.926, 122.926, 125.999),
        ("LIGHT_SIZE", 0, 1, 1),
    ]
    try:
        ws = agaveclient.AgaveClient('ws://localhost:1235/', protocols=['http-only', 'chat'])
        print("created client")

        def onGetInfo(jsondict):
            print(jsondict)

        def onOpen():
            ws.get_info("D:\\data\\april2019\\aligned_100s\\Interphase\\ACTB_36972_seg.ome.tif", onGetInfo)
            ws.render_frame(per_frame_commands, 1, "one")
            ws.render_frame(per_frame_commands2, 2, "two")

        ws.onOpened = onOpen
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        print("keyboard")
        ws.close()

