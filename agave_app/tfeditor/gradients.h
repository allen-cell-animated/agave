/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#pragma once
#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "AppScene.h"
#include "Histogram.h"

#include <QPushButton>
#include <QRadioButton>

class HoverPoints;
class ShadeWidget : public QWidget
{
  Q_OBJECT

public:
  enum ShadeType
  {
    RedShade,
    GreenShade,
    BlueShade,
    ARGBShade
  };

  ShadeWidget(const Histogram& histogram, ShadeType type, QWidget* parent);

  void setGradientStops(const QGradientStops& stops);

  void setEditable(bool editable);

  void paintEvent(QPaintEvent* e) override;

  QSize sizeHint() const override { return QSize(150, 40); }
  QPolygonF points() const;

  HoverPoints* hoverPoints() const { return m_hoverPoints; }

  uint colorAt(int x);

signals:
  void colorsChanged();

private:
  void generateShade();
  void drawHistogram(QPainter& p, int w, int h);

  ShadeType m_shade_type;
  QImage m_shade;
  HoverPoints* m_hoverPoints;
  QLinearGradient m_alpha_gradient;

  Histogram m_histogram;
};

class GradientEditor : public QWidget
{
  Q_OBJECT

public:
  GradientEditor(const Histogram& histogram, QWidget* parent = nullptr);

  void setControlPoints(const std::vector<LutControlPoint>& points);
  void setEditable(bool editable) { m_alpha_shade->setEditable(editable); }
public slots:
  void pointsUpdated();

signals:
  void gradientStopsChanged(const QGradientStops& stops);

private:
  ShadeWidget* m_alpha_shade;

protected:
  virtual void wheelEvent(QWheelEvent* event) override;
};

class GradientWidget : public QWidget
{
  Q_OBJECT

public:
  GradientWidget(const Histogram& histogram, GradientData* dataObject, QWidget* parent = nullptr);

public slots:
  void onGradientStopsChanged(const QGradientStops& stops);

signals:
  void gradientStopsChanged(const QGradientStops& stops);

private:
  void onSetWindowLevel(float window, float level);
  void onSetIsovalue(float isovalue, float width);
  void onSetHistogramPercentiles(float pctLow, float pctHigh);
  void forceDataUpdate();

  GradientEditor* m_editor;
  Histogram m_histogram;

  // owned externally, passed in via ctor
  GradientData* m_gradientData;
};

#endif // GRADIENTS_H
