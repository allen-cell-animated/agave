#pragma once

#include <QOpenGLShaderProgram>

#include <ome/files/Types.h>

#include <glm.h>


/**
    * 2D image shader program (simple, up to three channels).
    */
class GLImageShader2DnoLut : public QOpenGLShaderProgram
{

public:
    /**
    * Constructor.
    *
    * @param parent the parent of this object.
    */
    explicit GLImageShader2DnoLut();

    /// Destructor.
    ~GLImageShader2DnoLut();

    /// Enable vertex coordinates.
    void
    enableCoords();

    /// Disable vertex coordinates.
    void
    disableCoords();

    /**
    * Set vertex coordinates from array.
    *
    * @param offset data offset if using a buffer object otherwise
    * the coordinate values.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setCoords(const GLfloat *offset = 0,
            int            tupleSize = 2,
            int            stride = 0);

    /**
    * Set vertex coordinates from buffer object.
    *
    * @param coords the coordinate values; null if using a buffer object.
    * @param offset the offset into the coords buffer.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setCoords(GLuint  coords,
            const GLfloat  *offset = 0,
            int             tupleSize = 2,
            int             stride = 0);

    /// Enable texture coordinates.
    void
    enableTexCoords();

    /// Disable texture coordinates.
    void
    disableTexCoords();

    /**
    * Set texture coordinates from array.
    *
    * @param offset data offset if using a buffer object otherwise
    * the coordinate values.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setTexCoords(const GLfloat *offset = 0,
                int            tupleSize = 2,
                int            stride = 0);

    /**
    * Set texture coordinates from buffer object.
    *
    * @param coords the coordinate values; null if using a buffer
    * object.
    * @param offset the offset into the coords buffer.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setTexCoords(GLuint  coords,
                const GLfloat  *offset = 0,
                int             tupleSize = 2,
                int             stride = 0);

    /**
    * Set the texture to render.
    *
    * @param texunit the texture unit to use.
    */
    void
    setTexture(int texunit);

    /**
    * Set model view projection matrix.
    *
    * @param mvp the model view projection matrix.
    */
    void
    setModelViewProjection(const glm::mat4& mvp);

private:
    /// The vertex shader.
    QOpenGLShader *vshader;
    /// The fragment shader.
    QOpenGLShader *fshader;

    /// Vertex coordinates attribute.
    int attr_coords;
    /// Texture coordinates attribute.
    int attr_texcoords;
    /// Model view projection uniform.
    int uniform_mvp;
    /// Texture uniform.
    int uniform_texture;
};

