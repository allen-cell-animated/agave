#include "AppearanceDockWidget.h"

#include <QScrollArea>

QAppearanceWidget::QAppearanceWidget(QWidget* pParent,
                                     QRenderSettings* qrs,
                                     RenderSettings* rs,
                                     AreaLightObject* alo,
                                     SkyLightObject* slo,
                                     QAction* pToggleRotateAction,
                                     QAction* pToggleTranslateAction)
  : QWidget(pParent)
  , m_MainLayout()
  , m_AppearanceSettingsWidget(nullptr, qrs, rs, alo, slo, pToggleRotateAction, pToggleTranslateAction)
{
  // Create main layout
  m_MainLayout.setAlignment(Qt::AlignTop);
  setLayout(&m_MainLayout);

  QScrollArea* scrollArea = new QScrollArea();
  scrollArea->setWidgetResizable(true);
  scrollArea->setWidget(&m_AppearanceSettingsWidget);

  m_MainLayout.addWidget(scrollArea, 1, 0);
}

QAppearanceDockWidget::QAppearanceDockWidget(QWidget* parent,
                                             QRenderSettings* qrs,
                                             RenderSettings* rs,
                                             AreaLightObject* alo,
                                             SkyLightObject* slo,
                                             QAction* pToggleRotateAction,
                                             QAction* pToggleTranslateAction)
  : QDockWidget(parent)
  , m_VolumeAppearanceWidget(nullptr, qrs, rs, alo, slo, pToggleRotateAction, pToggleTranslateAction)
{
  setWindowTitle("Appearance");

  m_VolumeAppearanceWidget.setParent(this);

  setWidget(&m_VolumeAppearanceWidget);
}
