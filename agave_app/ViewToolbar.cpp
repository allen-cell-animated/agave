#include "ViewToolbar.h"

#include "renderlib/CCamera.h"

#include <QFile>
#include <QHBoxLayout>
#include <QIcon>
#include <QIconEngine>
#include <QPainter>
#include <QPixmap>
#include <QPixmapCache>
#include <QPushButton>
#include <QSvgRenderer>

ViewToolbar::ViewToolbar(QWidget* parent)
  : QWidget(parent)
{
  QHBoxLayout* toolbarLayout = new QHBoxLayout(this);
  toolbarLayout->setSpacing(0);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);

  static const int spacing = 4;

  homeButton = new QPushButton(QIcon(":/icons/Home-icon.svg"), "", this);
  homeButton->setToolTip(QString("<FONT>Reset view</FONT>"));
  homeButton->setStatusTip(tr("Reset the view"));
  homeButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  homeButton->adjustSize();
  homeButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(homeButton);

  frameViewButton = new QPushButton(QIcon(":/icons/frameView.svg"), "", this);
  frameViewButton->setToolTip(QString("<FONT>Frame view</FONT>"));
  frameViewButton->setStatusTip(tr("Frame the current view"));
  frameViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frameViewButton->adjustSize();
  frameViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frameViewButton);

  toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  orthoViewButton = new DualIconButton(QIcon(":/icons/perspView.svg"),
                                       QIcon(":/icons/orthoView.svg"),
                                       QString("<FONT>Switch to orthographic view</FONT>"),
                                       tr("Switch to orthographic view"),
                                       QString("<FONT>Switch to perspective view</FONT>"),
                                       tr("Switch to perspective view"),
                                       this);
  orthoViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  orthoViewButton->adjustSize();
  orthoViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(orthoViewButton);

  toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  topViewButton = new QPushButton(QIcon(":/icons/topView.svg"), "", this);
  topViewButton->setToolTip(QString("<FONT>Top view (-Y)</FONT>"));
  topViewButton->setStatusTip(tr("Set the view to look down the negative Y axis"));
  topViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  topViewButton->adjustSize();
  topViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(topViewButton);

  bottomViewButton = new QPushButton(QIcon(":/icons/bottomView.svg"), "", this);
  bottomViewButton->setToolTip(QString("<FONT>Bottom view (+Y)</FONT>"));
  bottomViewButton->setStatusTip(tr("Set the view to look down the positive Y axis"));
  bottomViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  bottomViewButton->adjustSize();
  bottomViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(bottomViewButton);

  frontViewButton = new QPushButton(QIcon(":/icons/frontView.svg"), "", this);
  frontViewButton->setToolTip(QString("<FONT>Front view (-Z)</FONT>"));
  frontViewButton->setStatusTip(tr("Set the view to look down the negative Z axis"));
  frontViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frontViewButton->adjustSize();
  frontViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frontViewButton);

  backViewButton = new QPushButton(QIcon(":/icons/backView.svg"), "", this);
  backViewButton->setToolTip(QString("<FONT>Back view (+Z)</FONT>"));
  backViewButton->setStatusTip(tr("Set the view to look down the positive Z axis"));
  backViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  backViewButton->adjustSize();
  backViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(backViewButton);

  leftViewButton = new QPushButton(QIcon(":/icons/leftView.svg"), "", this);
  leftViewButton->setToolTip(QString("<FONT>Left view (+X)</FONT>"));
  leftViewButton->setStatusTip(tr("Set the view to look down the positive X axis"));
  leftViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  leftViewButton->adjustSize();
  leftViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(leftViewButton);

  rightViewButton = new QPushButton(QIcon(":/icons/rightView.svg"), "", this);
  rightViewButton->setToolTip(QString("<FONT>Right view (-X)</FONT>"));
  rightViewButton->setStatusTip(tr("Set the view to look down the negative X axis"));
  rightViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  rightViewButton->adjustSize();
  rightViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(rightViewButton);

  toolbarLayout->addStretch();

  setLayout(toolbarLayout);
  setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
}

ViewToolbar::~ViewToolbar() {}

void
ViewToolbar::initFromCamera(const CCamera& camera)
{
  orthoViewButton->setState((camera.m_Projection == ProjectionMode::ORTHOGRAPHIC) ? 1 : 0);
}
