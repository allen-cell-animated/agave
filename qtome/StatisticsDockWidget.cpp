#include "StatisticsDockWidget.h"

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent)
  : QDockWidget(pParent)
  , m_MainLayout()
  , m_StatisticsWidget()
{
  setWindowTitle("Statistics");

  setWidget(&m_StatisticsWidget);
}
