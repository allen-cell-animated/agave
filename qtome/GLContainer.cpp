#include "GLContainer.h"

#include <QtWidgets>

GLContainer::GLContainer(QWidget *parent,
                            QWindow *window):
    QWidget(parent),
    m_window(window),
    m_child(QWidget::createWindowContainer(window))
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_child);
    setLayout(mainLayout);
}

GLContainer::~GLContainer()
{
}

QWindow *
GLContainer::getWindow() const
{
    return m_window;
}

QWidget *
GLContainer::getContainer() const
{
    return m_child;
}

