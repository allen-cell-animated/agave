set(SHADERS_CPP
"pathTraceVolume_frag_gen.hpp"
"pathTraceVolume_vert_gen.hpp"
"basicVolume_frag_gen.hpp"
)

# Add the generated C++ file to your target
target_sources(renderlib PRIVATE ${SHADERS_CPP})
