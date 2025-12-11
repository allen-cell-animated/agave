#pragma once
#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "AppScene.h"
#include "Controls.h"
#include "Histogram.h"
#include "qcustomplot.h"

#include <QPushButton>
#include <QRadioButton>

class GradientEditor : public QWidget
{
  Q_OBJECT

public:
  GradientEditor(const Histogram& histogram, QWidget* parent = nullptr);

  void setControlPoints(const std::vector<LutControlPoint>& points);
  void setEditMode(GradientEditMode gradientEditMode) { m_currentEditMode = gradientEditMode; }

  enum LockType
  {
    LockToNone = 0x00,
    LockToLeft = 0x01,
    LockToRight = 0x02,
    LockToTop = 0x04,
    LockToBottom = 0x08
  };

public slots:
  void onPlotMousePress(QMouseEvent* event);
  void onPlotMouseMove(QMouseEvent* event);
  void onPlotMouseRelease(QMouseEvent* event);
  void onPlotMouseDoubleClick(QMouseEvent* event);
  void onPlotMouseWheel(QWheelEvent* event);

signals:
  void gradientStopsChanged(const QGradientStops& stops);
  void interactivePointsChanged(float minIntensity, float maxIntensity);

private:
  Histogram m_histogram;

  GradientEditMode m_currentEditMode;

  QCustomPlot* m_customPlot;
  QCPBars* m_histogramBars;
  bool m_isDraggingPoint = false;
  int m_currentPointIndex = -1;

  QGradientStops buildStopsFromPlot();
  void set_shade_points(const QPolygonF& points, QCustomPlot* plot, const Histogram& histogram);

  QVector<uint32_t> m_locks;

protected:
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void changeEvent(QEvent* event) override;
};

class GradientWidget : public QWidget
{
  Q_OBJECT

public:
  GradientWidget(const Histogram& histogram, GradientData* dataObject, QWidget* parent = nullptr);

public slots:
  void onGradientStopsChanged(const QGradientStops& stops);
  void onInteractivePointsChanged(float minIntensity, float maxIntensity);

signals:
  void gradientStopsChanged(const QGradientStops& stops);

private:
  void onSetWindowLevel(float window, float level);
  void onSetIsovalue(float isovalue, float width);
  void onSetHistogramPercentiles(float pctLow, float pctHigh);
  void onSetMinMax(uint16_t minu16, uint16_t maxu16);
  void forceDataUpdate();

  GradientEditor* m_editor;
  Histogram m_histogram;

  // owned externally, passed in via ctor
  GradientData* m_gradientData;

  QIntSlider* minu16Slider = nullptr;
  QIntSlider* maxu16Slider = nullptr;
  QNumericSlider* windowSlider = nullptr;
  QNumericSlider* levelSlider = nullptr;
  QNumericSlider* isovalueSlider = nullptr;
  QNumericSlider* isorangeSlider = nullptr;
  QNumericSlider* pctLowSlider = nullptr;
  QNumericSlider* pctHighSlider = nullptr;
};

#endif // GRADIENTS_H
