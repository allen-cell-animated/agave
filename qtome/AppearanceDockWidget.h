#pragma once

#include <QtWidgets/QDockWidget>

//#include "PresetsWidget.h"
#include "AppearanceSettingsWidget.h"
//#include "TransferFunctionWidget.h"
//#include "NodeSelectionWidget.h"
//#include "NodePropertiesWidget.h"
//#include "TransferFunction.h"

class QAppearanceDockWidget;
class CScene;

class QAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QAppearanceWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, CScene* scene = nullptr);
	
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
    QAppearanceDockWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, CScene* scene = nullptr);

	void onNewImage(Scene* s) {
		m_VolumeAppearanceWidget.onNewImage(s); 
	}

protected:
	QAppearanceWidget		m_VolumeAppearanceWidget;
};