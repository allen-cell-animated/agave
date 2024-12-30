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
#convert_shader_to_cpp("basicVolume.frag" "basicVolume.frag")

find_package(Python3)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found")
endif()
set(SHADER_FILE "pathTraceVolume.frag" "basicVolume.frag")
set(OUTPUT_SHADER_FILE "pathTraceVolume_frag_gen.hpp" "basicVolume_frag_gen.hpp")
foreach(shader IN LISTS SHADER_FILE)
  message("Generate ${shader}...")
  execute_process(COMMAND ${Python3_EXECUTABLE} convert_shader.py ${shader} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# Add the generated C++ file to your target
target_sources(renderlib PRIVATE ${OUTPUT_SHADER_FILE})
