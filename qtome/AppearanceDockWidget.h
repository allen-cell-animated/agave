#pragma once

#include <QtWidgets/QDockWidget>

#include "AppearanceSettingsWidget.h"

class QAppearanceDockWidget;
class RenderSettings;

class QAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QAppearanceWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, RenderSettings* rs = nullptr);
	
	void onNewImage(Scene* s) {
		m_AppearanceSettingsWidget.onNewImage(s);
	}

public slots:
	void OnLoadPreset(const QString& Name);
	void OnSavePreset(const QString& Name);

protected:
	QGridLayout							m_MainLayout;
	//QPresetsWidget<QTransferFunction>	m_PresetsWidget;
	QAppearanceSettingsWidget			m_AppearanceSettingsWidget;
	//QTransferFunctionWidget				m_TransferFunctionWidget;
	//QNodeSelectionWidget				m_NodeSelectionWidget;
	//QNodePropertiesWidget				m_NodePropertiesWidget;
	
};

class QAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QAppearanceDockWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, RenderSettings* rs = nullptr);

	void onNewImage(Scene* s) {
		m_VolumeAppearanceWidget.onNewImage(s); 
	}

protected:
	QAppearanceWidget		m_VolumeAppearanceWidget;
};