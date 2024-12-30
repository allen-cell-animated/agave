function(convert_shader_to_cpp input_file output_file)
    # Read the shader file content
    file(READ "${input_file}" file_content)
    # Convert the file content to a list of strings, splitting by newlines
    string(REGEX REPLACE "\n" ";" file_content_list "${file_content}")

    # Print the list of strings
    # foreach(line IN LISTS file_content_list)
    #     message("${line}")
    # endforeach()

    # Escape special characters in the shader content
    #string(REGEX REPLACE "\\" "\\\\" shader_content "${shader_content}")
    #string(REGEX REPLACE "\"" "\\\"" shader_content "${shader_content}")
    #string(REGEX REPLACE "\n" "\\n" shader_content "${shader_content}")
    string(REPLACE "." "_" output_file "${output_file}")

    # Generate the C++ file content
    set(cpp_content "
#include <string>

const std::string ${output_file}_shader = R\"(
${file_content}
)\";

")

    # Write the C++ file
    file(WRITE "${output_file}_gen.hpp" "${cpp_content}")
endfunction()

# Convert your shader file
convert_shader_to_cpp("basicVolume.frag" "basicVolume.frag")
# convert_shader_to_cpp("pathTraceVolume.frag" "pathTraceVolume.frag")


set(SHADER_FILE "pathTraceVolume.frag")
set(OUTPUT_C_FILE "pathTraceVolume_frag_gen.hpp")
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/convert_shader.py ${SHADER_FILE} ${OUTPUT_C_FILE})
# Add a custom command to convert the shader
# add_custom_command(
#     OUTPUT ${OUTPUT_C_FILE}
#     COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/convert_shader.py ${SHADER_FILE} ${OUTPUT_C_FILE}
#     DEPENDS ${SHADER_FILE} ${CMAKE_SOURCE_DIR}/convert_shader.py
#     COMMENT "Converting ${SHADER_FILE} to C source file ${OUTPUT_C_FILE}"
# )

# Add the generated C++ file to your target
target_sources(renderlib PRIVATE 
basicVolume_frag_gen.hpp
pathTraceVolume_frag_gen.hpp
)
