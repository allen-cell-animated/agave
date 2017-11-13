#pragma once

#include <QtGui/QWindow>
#include <QtWidgets/QWidget>


/**
    * GL window container.
    *
    * A Qt GL window is an entire top-level window.  If this is to be
    * placed in a standard application window, it requires embedding
    * in a widget.  This widget serves to embed a GL window as a
    * regular widget.
    *
    * This may be used to embed any top-level window, not just GL
    * windows.  However, there's rarely any point for non-GL windows
    * since you can just contain their content directly.
    */
class GLContainer : public QWidget
{
    Q_OBJECT

public:
    /**
    * Create a window container.
    *
    * @param parent the parent of this object.
    * @param window the GL window to embed.
    */
    explicit GLContainer(QWidget *parent = 0,
                        QWindow *window = 0);

    /// Destructor.
    ~GLContainer();

    /**
    * Get contained GL window.
    *
    * @returns the GL window.
    */
    QWindow *
    getWindow() const;

    /**
    * Get child GL window container.
    *
    * @returns the container.
    */
    QWidget *
    getContainer() const;

private:
    /// GL window.
    QWindow *window;
    /// Child containing the GL window.
    QWidget *child;
};

