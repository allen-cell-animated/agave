#include "GLContainer.h"

#include <QtWidgets>

GLContainer::GLContainer(QWidget *parent,
                            QWindow *window):
    QWidget(parent),
    window(window),
    child(QWidget::createWindowContainer(window))
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(child);
    setLayout(mainLayout);
}

GLContainer::~GLContainer()
{
}

QWindow *
GLContainer::getWindow() const
{
    return window;
}

QWidget *
GLContainer::getContainer() const
{
    return child;
}

