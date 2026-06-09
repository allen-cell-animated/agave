#pragma once

#include "StatisticsWidget.h"

#include <QDockWidget>
#include <QGraphicsScene>
#include <QGridLayout>
#include <QWidget>
#include <QtGui>

class QStatisticsDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QStatisticsDockWidget(QWidget* pParent = nullptr);
  void setStatus(std::shared_ptr<CStatus> s) { m_StatisticsWidget.set(s); }

private:
  QWidget m_MainWidget;
  QGridLayout m_MainLayout;
  QStatisticsWidget m_StatisticsWidget;
  QGraphicsScene m_scene;
};
