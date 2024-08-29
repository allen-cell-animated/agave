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
#include <QToolBar>
#include <QWidgetAction>
#include <QMenu>

ViewToolbar::ViewToolbar(QWidget* parent)
  : QWidget(parent)
{
  QHBoxLayout* toolbarLayout = new QHBoxLayout(this);
  toolbarLayout->setSpacing(1);
  toolbarLayout->setContentsMargins(4, 4, 4, 4);

  static const int spacing = 8;

  homeButton = new QPushButton(QIcon(), "", this);
  homeButton->setObjectName("homeBtn");
  homeButton->setToolTip(QString("<FONT>Reset view</FONT>"));
  homeButton->setStatusTip(tr("Reset the view"));
  homeButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  homeButton->adjustSize();
  homeButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(homeButton);

  frameViewButton = new QPushButton(QIcon(), "", this);
  frameViewButton->setObjectName("frameViewBtn");
  frameViewButton->setToolTip(QString("<FONT>Frame view</FONT>"));
  frameViewButton->setStatusTip(tr("Frame the current view"));
  frameViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frameViewButton->adjustSize();
  frameViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frameViewButton);

  toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  orthoViewButton = new DualIconButton(QString("<FONT>Switch to orthographic view</FONT>"),
                                       tr("Switch to orthographic view"),
                                       QString("<FONT>Switch to perspective view</FONT>"),
                                       tr("Switch to perspective view"),
                                       this);
  orthoViewButton->setObjectName("orthoViewBtn");
  orthoViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  orthoViewButton->adjustSize();
  orthoViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(orthoViewButton);

  // toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  topViewButton = new QPushButton(QIcon(), "", this);
  topViewButton->setObjectName("topViewBtn");
  topViewButton->setToolTip(QString("<FONT>Top view (-Y)</FONT>"));
  topViewButton->setStatusTip(tr("Set the view to look down the negative Y axis"));
  topViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  topViewButton->adjustSize();
  topViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(topViewButton);

  bottomViewButton = new QPushButton(QIcon(), "", this);
  bottomViewButton->setObjectName("bottomViewBtn");
  bottomViewButton->setToolTip(QString("<FONT>Bottom view (+Y)</FONT>"));
  bottomViewButton->setStatusTip(tr("Set the view to look down the positive Y axis"));
  bottomViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  bottomViewButton->adjustSize();
  bottomViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(bottomViewButton);

  frontViewButton = new QPushButton(QIcon(), "", this);
  frontViewButton->setObjectName("frontViewBtn");
  frontViewButton->setToolTip(QString("<FONT>Front view (-Z)</FONT>"));
  frontViewButton->setStatusTip(tr("Set the view to look down the negative Z axis"));
  frontViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frontViewButton->adjustSize();
  frontViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(frontViewButton);

  backViewButton = new QPushButton(QIcon(), "", this);
  backViewButton->setObjectName("backViewBtn");
  backViewButton->setToolTip(QString("<FONT>Back view (+Z)</FONT>"));
  backViewButton->setStatusTip(tr("Set the view to look down the positive Z axis"));
  backViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  backViewButton->adjustSize();
  backViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(backViewButton);

  leftViewButton = new QPushButton(QIcon(), "", this);
  leftViewButton->setObjectName("leftViewBtn");
  leftViewButton->setToolTip(QString("<FONT>Left view (+X)</FONT>"));
  leftViewButton->setStatusTip(tr("Set the view to look down the positive X axis"));
  leftViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  leftViewButton->adjustSize();
  leftViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(leftViewButton);

  rightViewButton = new QPushButton(QIcon(), "", this);
  rightViewButton->setObjectName("rightViewBtn");
  rightViewButton->setToolTip(QString("<FONT>Right view (-X)</FONT>"));
  rightViewButton->setStatusTip(tr("Set the view to look down the negative X axis"));
  rightViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  rightViewButton->adjustSize();
  rightViewButton->setFocusPolicy(Qt::NoFocus);
  // toolbarLayout->addWidget(rightViewButton);

  toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  axisHelperButton = new QPushButton(QIcon(), "", this);
  axisHelperButton->setObjectName("axisHelperBtn");
  axisHelperButton->setToolTip(QString("<FONT>Show/Hide axis helper</FONT>"));
  axisHelperButton->setStatusTip(tr("Show/Hide axis helper"));
  axisHelperButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  axisHelperButton->adjustSize();
  axisHelperButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(axisHelperButton);

  toolbarLayout->addItem(new QSpacerItem(spacing, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

  QMenu* menu = new QMenu("QuickViews", this);
  QToolBar* toolbar = new QToolBar();
  toolbar->addWidget(rightViewButton);
  toolbar->addWidget(leftViewButton);
  toolbar->addWidget(frontViewButton);
  toolbar->addWidget(backViewButton);
  toolbar->addWidget(topViewButton);
  toolbar->addWidget(bottomViewButton);
  QWidgetAction* act = new QWidgetAction(toolbar);
  act->setDefaultWidget(toolbar);
  menu->addAction(act);

  homeButton0 = new QPushButton(QIcon(), "", this);
  homeButton0->setObjectName("homeBtn");
  homeButton0->setToolTip(QString("<FONT>Quick Views</FONT>"));
  homeButton0->setStatusTip(tr("Quickly set an axis-aligned view"));
  homeButton0->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  homeButton0->adjustSize();
  homeButton0->setFocusPolicy(Qt::NoFocus);
  // homeButton0->setMenu(menu);
  toolbarLayout->addWidget(homeButton0);
  connect(homeButton0, &QPushButton::clicked, [menu, this]() {
    menu->exec(homeButton0->mapToGlobal(homeButton0->rect().bottomLeft()));
  });

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
