function(convert_shader_to_cpp input_file output_file)
    # Read the shader file content
    file(READ "${input_file}" shader_content)

    # Escape special characters in the shader content
    #string(REGEX REPLACE "\\" "\\\\" shader_content "${shader_content}")
    #string(REGEX REPLACE "\"" "\\\"" shader_content "${shader_content}")
    #string(REGEX REPLACE "\n" "\\n" shader_content "${shader_content}")
    string(REPLACE "." "_" output_file "${output_file}")

    # Generate the C++ file content
    set(cpp_content "
#include <string>

const std::string ${output_file}_shader = R\"(
${shader_content}
)\";

")

    # Write the C++ file
    file(WRITE "${output_file}_gen.hpp" "${cpp_content}")
endfunction()

# Convert your shader file
convert_shader_to_cpp("shader.frag" "shader")

# Add the generated C++ file to your target
target_sources(renderlib PRIVATE shader_gen.hpp)
