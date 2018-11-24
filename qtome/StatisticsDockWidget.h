#pragma once

#include <QtGui>
#include <QGraphicsScene>

#include "StatisticsWidget.h"

class QStatisticsDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QStatisticsDockWidget(QWidget* pParent = 0);
	void setStatus(CStatus* s) {
		m_StatisticsWidget.set(s);
	}

private:
	QGridLayout			m_MainLayout;
	QStatisticsWidget	m_StatisticsWidget;
	QGraphicsScene m_scene;
};