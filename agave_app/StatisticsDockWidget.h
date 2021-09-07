#pragma once

#include <QDockWidget>
#include <QGraphicsScene>
#include <QtGui>

#include "StatisticsWidget.h"

class QStatisticsDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  QStatisticsDockWidget(QWidget* pParent = 0);
  void setStatus(std::shared_ptr<CStatus> s) { m_StatisticsWidget.set(s); }

private:
  QGridLayout m_MainLayout;
  QStatisticsWidget m_StatisticsWidget;
  QGraphicsScene m_scene;
};
