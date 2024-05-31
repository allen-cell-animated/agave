#include "ViewToolbar.h"

#include <QFile>
#include <QHBoxLayout>
#include <QIcon>
#include <QIconEngine>
#include <QPainter>
#include <QPixmap>
#include <QPixmapCache>
#include <QPushButton>
#include <QSvgRenderer>

class DualIconButton : public QPushButton
{
private:
  QIcon icon1;
  QIcon icon2;
  QString tooltip1;
  QString tooltip2;
  QString statustip1;
  QString statustip2;
  int state;

public:
  DualIconButton(const QIcon& icon1,
                 const QIcon& icon2,
                 const QString& tooltip1,
                 const QString& statustip1,
                 const QString& tooltip2,
                 const QString& statustip2,
                 QWidget* parent = nullptr)
    : icon1(icon1)
    , icon2(icon2)
    , tooltip1(tooltip1)
    , tooltip2(tooltip2)
    , statustip1(statustip1)
    , statustip2(statustip2)
    , state(0)
    , QPushButton(parent)
  {

    setIcon(icon1);
    setToolTip(tooltip1);
    setStatusTip(statustip1);
    connect(this, &QPushButton::clicked, this, &DualIconButton::toggleIcon);
  }

  void toggleIcon()
  {
    state = 1 - state;
    if (state == 1) {
      setIcon(icon2);
      setToolTip(tooltip2);
      setStatusTip(statustip2);
    } else {
      setIcon(icon1);
      setToolTip(tooltip1);
      setStatusTip(statustip1);
    }
  }
};

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

  orthoViewButton = new DualIconButton(QIcon(":/icons/orthoView.svg"),
                                       QIcon(":/icons/perspView.svg"),
                                       QString("<FONT>Ortho view</FONT>"),
                                       tr("Toggle perspective and orthographic camera projection modes"),
                                       QString("<FONT>Persp view</FONT>"),
                                       tr("Toggle perspective and orthographic camera projection modes"),
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
