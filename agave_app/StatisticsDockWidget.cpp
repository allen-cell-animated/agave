#include "StatisticsDockWidget.h"

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent)
  : QDockWidget(pParent)
  , m_MainWidget(this)
  , m_MainLayout()
  , m_StatisticsWidget(&m_MainWidget)
{
  setWindowTitle("Statistics");

  m_MainLayout.setContentsMargins(0, 0, 8, 0);
  m_MainLayout.setSpacing(0);
  m_MainWidget.setLayout(&m_MainLayout);
  m_MainLayout.addWidget(&m_StatisticsWidget, 0, 0);

  setWidget(&m_MainWidget);
}
