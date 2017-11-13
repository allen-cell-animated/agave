#pragma once

#include <QOpenGLShaderProgram>

#include <glm.h>

/**
    * 2D flat (solid fill) shader program.
    */
class GLFlatShader2D : public QOpenGLShaderProgram
{

public:
    /**
    * Constructor.
    *
    * @param parent the parent of this object.
    */
    explicit GLFlatShader2D();

    /// Destructor.
    ~GLFlatShader2D();

    /// @copydoc GLImageShader2D::enableCoords()
    void
    enableCoords();

    /// @copydoc GLImageShader2D::enableCoords()
    void
    disableCoords();

    /// @copydoc GLImageShader2D::setCoords(const GLfloat*, int, int)
    void
    setCoords(const GLfloat *offset,
            int            tupleSize,
            int            stride = 0);

    /// @copydoc GLImageShader2D::setCoords(QOpenGLBuffer&, const GLfloat*, int, int)
    void
    setCoords(GLuint  coords,
            const GLfloat  *offset,
            int             tupleSize,
            int             stride = 0);

    /**
    * Set fill colour.
    *
    * @param colour the RGBA fill colour.
    */
    void
    setColour(const glm::vec4& colour);

    /**
    * Set xy offset in model space.
    *
    * @param offset the offset to apply to the model.
    */
    void
    setOffset(const glm::vec2& offset);

    /// @copydoc GLImageShader2D::setModelViewProjection(const glm::mat4& mvp)
    void
    setModelViewProjection(const glm::mat4& mvp);

private:
    /// @copydoc GLImageShader2D::vshader
    QOpenGLShader *vshader;
    /// @copydoc GLImageShader2D::fshader
    QOpenGLShader *fshader;

    /// @copydoc GLImageShader2D::attr_coords
    int attr_coords;
    /// Fill colour uniform.
    int uniform_colour;
    /// Model offset uniform.
    int uniform_offset;
    /// @copydoc GLImageShader2D::uniform_mvp
    int uniform_mvp;
};
