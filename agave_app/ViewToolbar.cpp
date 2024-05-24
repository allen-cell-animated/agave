#include "ViewToolbar.h"

#include <QHBoxLayout>
#include <QIcon>
#include <QPixmap>
#include <QPushButton>

ViewToolbar::ViewToolbar(QWidget* parent)
  : QWidget(parent)
{
  QHBoxLayout* toolbarLayout = new QHBoxLayout(this);
  toolbarLayout->setSpacing(0);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);

  topViewButton = new QPushButton(QIcon(":/icons/topView.svg"), "", this);
  topViewButton->setToolTip(QString("<FONT>Top view</FONT>"));
  topViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  topViewButton->adjustSize();
  topViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(topViewButton);

  frontViewButton = new QPushButton(QIcon(":/icons/frontView.svg"), "", this);
  frontViewButton->setToolTip(QString("<FONT>Front view</FONT>"));
  frontViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frontViewButton->adjustSize();
  frontViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frontViewButton);

  sideViewButton = new QPushButton(QIcon(":/icons/leftView.svg"), "", this);
  sideViewButton->setToolTip(QString("<FONT>Side view</FONT>"));
  sideViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  sideViewButton->adjustSize();
  sideViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(sideViewButton);

  //   int width = zoomFitButton->fontMetrics().boundingRect(zoomFitButton->text()).width() * 2 + 7;
  //   zoomInButton->setMaximumWidth(width);
  //   zoomOutButton->setMaximumWidth(width);
  //   zoomFitButton->setMaximumWidth(width);

  toolbarLayout->addStretch();

  setLayout(toolbarLayout);
  setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  // move(0, 0);
}

ViewToolbar::~ViewToolbar() {}

void
ViewToolbar::positionToolbar()
{
  // float over parent or lay out in a vboxlayout?
  // auto s = parent()->size();
  // move(s.width() - width() - TOOLBAR_INSET, s.height() - height() - TOOLBAR_INSET);
  // move(0, 0);
}
