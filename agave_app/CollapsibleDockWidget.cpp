#include "CollapsibleDockWidget.h"

#include <assert.h>
#include <iostream>

#include <QPointer>
#include <QSizePolicy>
#include <QTimer>
#include <QTreeView>

CollapsibleDockWidget::CollapsibleDockWidget(const QString& title, QWidget* parent, Qt::WindowFlags flags)
  : QDockWidget(title, parent, flags)
{
  CollapsibleDockWidget::TitleBar* titleBar = new CollapsibleDockWidget::TitleBar(this);
  setTitleBarWidget(titleBar);
}

CollapsibleDockWidget::CollapsibleDockWidget(QWidget* parent, Qt::WindowFlags flags)
  : QDockWidget(parent, flags)
{
  CollapsibleDockWidget::TitleBar* titleBar = new CollapsibleDockWidget::TitleBar(this);
  setTitleBarWidget(titleBar);
}

void
CollapsibleDockWidget::setCollapsibleWidget(QWidget* w)
{
  CollapsibleDockWidget::InnerWidgetWrapper* wid = new CollapsibleDockWidget::InnerWidgetWrapper(this);
  wid->setWidget(w);
  QDockWidget::setWidget(wid);

  if (isVisible()) {
    widget()->show();
  }
}

void
CollapsibleDockWidget::setCollapsed(bool collapsed)
{
  CollapsibleDockWidget::InnerWidgetWrapper* innerWidget =
    dynamic_cast<CollapsibleDockWidget::InnerWidgetWrapper*>(widget());

  if (innerWidget != NULL) {
    innerWidget->setCollapsed(collapsed);
  } else {
    std::cerr << "DockWidget is not collapsible" << std::endl;
  }
}

bool
CollapsibleDockWidget::isCollapsed()
{
  CollapsibleDockWidget::InnerWidgetWrapper* innerWidget =
    dynamic_cast<CollapsibleDockWidget::InnerWidgetWrapper*>(widget());
  return innerWidget != NULL ? innerWidget->isCollapsed() : false;
}

void
CollapsibleDockWidget::toggleCollapsed()
{
  setCollapsed(!isCollapsed());
}

void
CollapsibleDockWidget::windowTitleChanged()
{
  CollapsibleDockWidget::TitleBar* titleBar = dynamic_cast<CollapsibleDockWidget::TitleBar*>(this->titleBarWidget());

  if (titleBar) {
    titleBar->windowTitleChanged();
  }
}

CollapsibleDockWidget::InnerWidgetWrapper::InnerWidgetWrapper(QDockWidget* parent)
  : QWidget(parent)
  , widget(NULL)
  , hlayout(new QHBoxLayout(this))
  , widget_height(0)
  , oldSize(0, 0)
{
  this->hlayout->setSpacing(0);
  this->hlayout->setContentsMargins(0, 0, 0, 0);
  this->setLayout(this->hlayout);
  QDockWidget* parentDockWidget = dynamic_cast<QDockWidget*>(parent);
  assert(parentDockWidget != NULL);
  oldMinimumSizeParent = parentDockWidget->minimumSize();
  oldMaximumSizeParent = parentDockWidget->maximumSize();
  oldMinimumSize = minimumSize();
  oldMaximumSize = maximumSize();
}

void
CollapsibleDockWidget::InnerWidgetWrapper::setWidget(QWidget* widget)
{
  this->widget = widget;
  this->widget_height = widget->height();
  this->layout()->addWidget(widget);
  this->oldSize = this->size();
  QDockWidget* parentDockWidget = dynamic_cast<QDockWidget*>(this->parent());
  assert(parentDockWidget != NULL);
  oldMinimumSizeParent = parentDockWidget->minimumSize();
  oldMaximumSizeParent = parentDockWidget->maximumSize();
  oldMinimumSize = minimumSize();
  oldMaximumSize = maximumSize();
}

bool
CollapsibleDockWidget::InnerWidgetWrapper::isCollapsed()
{
  return !this->widget->isVisible();
}

void
CollapsibleDockWidget::InnerWidgetWrapper::setCollapsed(bool collapsed)
{
  QDockWidget* parentDockWidget = dynamic_cast<QDockWidget*>(this->parent());
  assert(parentDockWidget != NULL);
  CollapsibleDockWidget::TitleBar* parentDockWidgetTitleBar =
    dynamic_cast<CollapsibleDockWidget::TitleBar*>(parentDockWidget->titleBarWidget());
  assert(parentDockWidgetTitleBar != NULL);

  if (!collapsed) {
    parentDockWidget->setMinimumSize(oldMinimumSizeParent);
    parentDockWidget->setMaximumSize(oldMaximumSizeParent);
    this->widget->show();
    parentDockWidgetTitleBar->showTitle(true);
    parentDockWidgetTitleBar->setCollapsed(false);
    this->layout()->addWidget(this->widget);
    this->setMinimumSize(oldMinimumSize);
    this->setMaximumSize(oldMaximumSize);
    this->setBaseSize(this->oldSize);
    this->resize(this->oldSize);
  } else {
    this->oldSize = this->size();
    oldMinimumSizeParent = parentDockWidget->minimumSize();
    oldMaximumSizeParent = parentDockWidget->maximumSize();
    oldMinimumSize = minimumSize();
    oldMaximumSize = maximumSize();
    this->layout()->removeWidget(this->widget);
    this->widget->hide();
    parentDockWidgetTitleBar->showTitle(true);
    parentDockWidgetTitleBar->setCollapsed(true);
    parentDockWidget->setMinimumSize(oldMinimumSizeParent.width(), 25);
    parentDockWidget->setMaximumSize(oldMaximumSizeParent.width(), 25);
    QTimer::singleShot(1, parentDockWidget, SLOT(setCollapsedSizes()));
  }
}

void
CollapsibleDockWidget::setCollapsedSizes()
{
  CollapsibleDockWidget::InnerWidgetWrapper* innerWidget =
    dynamic_cast<CollapsibleDockWidget::InnerWidgetWrapper*>(widget());
  assert(innerWidget != NULL);
  setMinimumSize(25, 25);

  if (features() & QDockWidget::DockWidgetVerticalTitleBar) {
    setMaximumSize(25, innerWidget->getOldMaximumSizeParent().height());
  } else {
    setMaximumSize(innerWidget->getOldMaximumSizeParent().width(), 25);
  }
}

CollapsibleDockWidget::TitleBar::TitleBar(QWidget* parent)
  : QWidget(parent)
  , hlayout(new QHBoxLayout(this))
  , collapse(new QPushButton(this))
  , close(new QPushButton(this))
  , undock(new QPushButton(this))
  , title(new QLabel(parent->windowTitle()))
{
  this->hlayout->setDirection(QBoxLayout::Direction::LeftToRight);
  this->hlayout->setSpacing(0);
  this->hlayout->setContentsMargins(0, 0, 0, 0);
  this->setLayout(this->hlayout);
  this->hlayout->addWidget(collapse);
  collapse->setIcon(QIcon(":/images/branch-open.png"));
  collapse->setCheckable(false);
  collapse->setFixedSize(20, 20);
  connect(collapse, SIGNAL(released()), parent, SLOT(toggleCollapsed()));
  this->hlayout->addStretch();
  this->hlayout->addWidget(title);
  this->hlayout->addStretch();
  this->hlayout->addWidget(undock);
  undock->setIcon(QIcon(":/icons/unlinked.png"));
  undock->setCheckable(false);
  undock->setFixedSize(20, 20);
  QObject::connect(undock, &QPushButton::released, [this]() {
    QDockWidget* parentDockWidget = dynamic_cast<QDockWidget*>(this->parent());
    parentDockWidget->setFloating(true);
  });
  this->hlayout->addWidget(close);
  close->setIcon(QIcon(":/icons/lock.png"));
  close->setCheckable(false);
  close->setFixedSize(20, 20);
  connect(close, SIGNAL(released()), parent, SLOT(close()));
}

void
CollapsibleDockWidget::TitleBar::windowTitleChanged()
{
  QDockWidget* parentDockWidget = dynamic_cast<QDockWidget*>(this->parent());
  title->setText(parentDockWidget->windowTitle());
}

void
CollapsibleDockWidget::TitleBar::showTitle(bool show)
{
  title->setVisible(show);

  if (show) {
    collapse->setIcon(QIcon(":/icons/branch-open.png"));
  } else {
    collapse->setIcon(QIcon(":/icons/branch-closed.png"));
  }
}

void
CollapsibleDockWidget::TitleBar::setCollapsed(bool collapsed)
{
  if (collapsed) {
    collapse->setIcon(QIcon(":/icons/branch-open.png"));
  } else {
    collapse->setIcon(QIcon(":/icons/branch-closed.png"));
  }
}

QSize const&
CollapsibleDockWidget::InnerWidgetWrapper::getOldMaximumSizeParent() const
{
  return oldMaximumSizeParent;
}
