#include "gradients.h"

#include "Controls.h"
#include "qcustomplot.h"
#include "renderlib/Defines.h"
#include "renderlib/Logging.h"
#include "renderlib/MathUtil.h"

#include <algorithm>

std::vector<LutControlPoint>
gradientStopsToVector(QGradientStops& stops)
{
  std::vector<LutControlPoint> v;
  for (int i = 0; i < stops.size(); ++i) {
    v.push_back(LutControlPoint(stops.at(i).first, stops.at(i).second.alphaF()));
  }
  return v;
}

QGradientStops
vectorToGradientStops(std::vector<LutControlPoint>& v)
{
  QGradientStops stops;
  for (int i = 0; i < v.size(); ++i) {
    stops.push_back(
      QPair<qreal, QColor>(v[i].first, QColor::fromRgbF(v[i].second, v[i].second, v[i].second, v[i].second)));
  }
  return stops;
}

static void
bound_point(double x, double y, const QRectF& bounds, int lock, double& out_x, double& out_y)
{
  qreal left = bounds.left();
  qreal right = bounds.right();
  // notice top/bottom switch here.
  qreal bottom = bounds.top();
  qreal top = bounds.bottom();

  out_x = x;
  out_y = y;

  if (x <= left || (lock & GradientEditor::LockToLeft))
    out_x = left;
  else if (x >= right || (lock & GradientEditor::LockToRight))
    out_x = right;

  if (y >= top || (lock & GradientEditor::LockToTop))
    out_y = top;
  else if (y <= bottom || (lock & GradientEditor::LockToBottom))
    out_y = bottom;
}

static constexpr double SCATTERSIZE = 10.0;

GradientEditor::GradientEditor(const Histogram& histogram, QWidget* parent)
  : QWidget(parent)
  , m_histogram(histogram)
{
  QVBoxLayout* vbox = new QVBoxLayout(this);
  vbox->setSpacing(1);
  // vbox->setMargin(1);

  m_customPlot = new QCustomPlot(this);

  // first graph will be histogram
  QPalette pal = m_customPlot->palette();
  QColor histFillColor = pal.color(QPalette::Link).lighter(150);
  m_histogramBars = new QCPBars(m_customPlot->xAxis, m_customPlot->yAxis);
  QPen barPen = m_histogramBars->pen();
  barPen.setColor(histFillColor);
  m_histogramBars->setPen(barPen);
  m_histogramBars->setWidthType(QCPBars::wtPlotCoords);
  float firstBinCenter, lastBinCenter, binSize;
  histogram.bin_range(histogram._bins.size(), firstBinCenter, lastBinCenter, binSize);
  m_histogramBars->setWidth(binSize);
  QVector<double> keyData;
  QVector<double> valueData;
  static constexpr double MIN_BAR_HEIGHT = 0.01; // Minimum height for nonzero bins (0.1% of max)
  for (size_t i = 0; i < histogram._bins.size(); ++i) {
    keyData << firstBinCenter + i * binSize;
    if (histogram._bins[i] == 0) {
      // Zero bins get zero height
      valueData << 0.0;
    } else {
      // Nonzero bins get at least the minimum height
      double normalizedHeight = (double)histogram._bins[i] / (double)histogram._bins[histogram._maxBin];
      valueData << std::max(normalizedHeight, MIN_BAR_HEIGHT);
    }
  }
  m_histogramBars->setData(keyData, valueData);
  m_histogramBars->setSelectable(QCP::stNone);

  // first "graph" will the the piecewise linear transfer function
  m_customPlot->addGraph();
  m_customPlot->graph(0)->setPen(QPen(Qt::black)); // line color blue for first graph
  QPen scatterPen(Qt::black);
  scatterPen.setWidthF(1.0);
  // QBrush scatterBrush(Qt::white);
  m_customPlot->graph(0)->setScatterStyle(
    QCPScatterStyle(QCPScatterStyle::ssCircle, scatterPen, Qt::NoBrush, SCATTERSIZE));
  m_customPlot->graph(0)->setSelectable(QCP::stSingleData);

  //   give the axes some labels:
  m_customPlot->xAxis->setLabel("");
  m_customPlot->yAxis->setLabel("");

  // set axes ranges, so we see all data:
  m_customPlot->xAxis->setRange(histogram._dataMin, histogram._dataMax);
  m_customPlot->xAxis->ticker()->setTickCount(4);
  m_customPlot->xAxis->ticker()->setTickOrigin(histogram._dataMin);
  auto tickLabelFont = m_customPlot->xAxis->tickLabelFont();
  tickLabelFont.setPointSize((float)tickLabelFont.pointSize() * 0.75);
  m_customPlot->xAxis->setTickLabelFont(tickLabelFont);
  QPen penx = m_customPlot->xAxis->basePen();
  penx.setWidthF(1.0);
  m_customPlot->xAxis->setBasePen(penx);
  static constexpr double AXIS_OFFSET_FRACTION = 0.1;
  m_customPlot->xAxis->setOffset(AXIS_OFFSET_FRACTION);

  m_customPlot->yAxis->setRange(0 - AXIS_OFFSET_FRACTION, 1 + AXIS_OFFSET_FRACTION);
  m_customPlot->yAxis->ticker()->setTickCount(1);
  tickLabelFont = m_customPlot->yAxis->tickLabelFont();
  tickLabelFont.setPointSize((float)tickLabelFont.pointSize() * 0.75);
  m_customPlot->yAxis->setTickLabelFont(tickLabelFont);
  QPen peny = m_customPlot->yAxis->basePen();
  peny.setWidthF(1.0);
  m_customPlot->yAxis->setBasePen(peny);

  m_customPlot->xAxis->grid()->setVisible(true);
  m_customPlot->xAxis->grid()->setSubGridVisible(true);
  m_customPlot->yAxis->grid()->setVisible(true);
  m_customPlot->yAxis->grid()->setSubGridVisible(true);

  m_customPlot->setInteractions(
    QCP::iRangeDrag | QCP::iRangeZoom |
    QCP::iSelectPlottables); // allow user to drag axis ranges with mouse, zoom with mouse wheel
  m_customPlot->axisRect()->setRangeDrag(Qt::Horizontal);
  m_customPlot->axisRect()->setRangeZoom(Qt::Horizontal);
  // m_customPlot->axisRect()->setMargins(QMargins(10, 10, 10, 10));

  m_customPlot->replot();

  connect(m_customPlot, &QCustomPlot::mousePress, this, &GradientEditor::onPlotMousePress);
  connect(m_customPlot, &QCustomPlot::mouseMove, this, &GradientEditor::onPlotMouseMove);
  connect(m_customPlot, &QCustomPlot::mouseRelease, this, &GradientEditor::onPlotMouseRelease);
  connect(m_customPlot, &QCustomPlot::mouseWheel, this, &GradientEditor::onPlotMouseWheel);
  connect(m_customPlot, &QCustomPlot::mouseDoubleClick, this, &GradientEditor::onPlotMouseDoubleClick);

  vbox->addWidget(m_customPlot);
}

void
GradientEditor::changeEvent(QEvent* event)
{
  // this might be too many but ThemeChange only seems to work on the QMainWindow
  if (event->type() == QEvent::ThemeChange || event->type() == QEvent::ApplicationPaletteChange ||
      event->type() == QEvent::StyleChange || event->type() == QEvent::PaletteChange) {
    // check for dark or light mode
    auto sh = QGuiApplication::styleHints();
    auto colorScheme = sh->colorScheme();
    if (colorScheme == Qt::ColorScheme::Dark) {
      LOG_DEBUG << "Switching gradient editor histogram to dark mode";
      m_customPlot->graph(0)->setPen(QPen(Qt::lightGray)); // line color blue for first graph
      QPen scatterPen(Qt::lightGray);
      scatterPen.setWidthF(1.0);
      // QBrush scatterBrush(Qt::white);
      m_customPlot->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssCircle, scatterPen, Qt::NoBrush, SCATTERSIZE));
      m_customPlot->replot();
    } else if (colorScheme == Qt::ColorScheme::Light) {
      LOG_DEBUG << "Switching gradient editor histogram to light mode";
      m_customPlot->graph(0)->setPen(QPen(Qt::black)); // line color blue for first graph
      QPen scatterPen(Qt::black);
      scatterPen.setWidthF(1.0);
      // QBrush scatterBrush(Qt::white);
      m_customPlot->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssCircle, scatterPen, Qt::NoBrush, SCATTERSIZE));
      m_customPlot->replot();
    }
  }
  QWidget::changeEvent(event);
}

void
GradientEditor::onPlotMousePress(QMouseEvent* event)
{
  // in custom mode, any click is either ON a point or creating a new point?
  bool isCustomMode = (m_currentEditMode == GradientEditMode::CUSTOM);
  bool isMinMaxMode = (m_currentEditMode == GradientEditMode::MINMAX);
  bool isWindowLevelMode = (m_currentEditMode == GradientEditMode::WINDOW_LEVEL);
  bool isPercentileMode = (m_currentEditMode == GradientEditMode::PERCENTILE);
  bool isInteractiveMode = isCustomMode || isMinMaxMode || isWindowLevelMode || isPercentileMode;

  if (!isInteractiveMode) {
    return;
  }

  // let's look to see if a data point was clicked.

  int indexOfDataPoint = -1;
  double dist = 1E+9;

  auto graph = m_customPlot->graph(0);
  for (int n = 0; n < (graph->data()->size()); n++) {
    // get xy of each data pt in pixels. compare with scattersize.
    // first hit wins.
    double x = (graph->data()->begin() + n)->key;
    double y = (graph->data()->begin() + n)->value;
    double px = m_customPlot->xAxis->coordToPixel(x);
    double py = m_customPlot->yAxis->coordToPixel(y);
    double dx = (px - (double)event->pos().x());
    double dy = (py - (double)event->pos().y());
    dist = sqrt(dx * dx + dy * dy);
    if (dist < SCATTERSIZE / 2.0) {
      indexOfDataPoint = n;
      // remember dist!
      break;
    }
  }

  if (event->button() == Qt::LeftButton) {

    // if we didn't click on a point, then we could add a point (only in custom mode):
    if (indexOfDataPoint == -1 && isCustomMode) {
      // this checks to see if user clicked along the line anywhere close.
      QCPGraph* plottable = m_customPlot->plottableAt<QCPGraph>(event->pos(), true, &indexOfDataPoint);
      if (plottable != nullptr && indexOfDataPoint > -1) {
        // create a new point at x, y
        double x = m_customPlot->xAxis->pixelToCoord(event->pos().x());
        double y = m_customPlot->yAxis->pixelToCoord(event->pos().y());
        // find first point above x to know the index?
        for (int n = 0; n < (graph->data()->size()); n++) {
          // get xy of each data pt in pixels. compare with scattersize.
          // first hit wins.
          double xn = (graph->data()->begin() + n)->key;
          if (x < xn) {
            // the index of x will be n.
            indexOfDataPoint = n;
            break;
          }
        }
        m_locks.insert(indexOfDataPoint, 0);
        graph->addData(x, y);
        graph->data()->sort();
        m_customPlot->replot();
      }
    }

    if (indexOfDataPoint > -1) {
      // In MINMAX, Window/Level, and Percentile modes, only allow dragging of second and third points (threshold
      // points)
      if (isMinMaxMode || isWindowLevelMode || isPercentileMode) {
        int dataSize = graph->data()->size();
        if (indexOfDataPoint != 1 && indexOfDataPoint != 2) {
          // Not the second or third point (threshold points), don't allow dragging
          return;
        }
      }

      m_isDraggingPoint = true;
      m_currentPointIndex = indexOfDataPoint;
      // turn off axis dragging while we are dragging a point
      m_customPlot->axisRect()->setRangeDrag((Qt::Orientations)0);
      // swallow this event so it doesn't propagate to the plot
      event->accept();
    }
  } else if (event->button() == Qt::RightButton) {
    // Only allow point deletion in custom mode
    if (indexOfDataPoint >= 0 && isCustomMode) {
      if (m_locks[indexOfDataPoint] == 0) {
        m_locks.remove(indexOfDataPoint);
        // remove data pt from plot
        graph->data()->remove((graph->data()->begin() + indexOfDataPoint)->key);

        // update the stuff because we did something
        emit gradientStopsChanged(this->buildStopsFromPlot());
        m_customPlot->replot();
        event->accept();
      }
    }
  }
}
void
GradientEditor::onPlotMouseMove(QMouseEvent* event)
{
  bool isCustomMode = (m_currentEditMode == GradientEditMode::CUSTOM);
  bool isMinMaxMode = (m_currentEditMode == GradientEditMode::MINMAX);
  bool isWindowLevelMode = (m_currentEditMode == GradientEditMode::WINDOW_LEVEL);
  bool isPercentileMode = (m_currentEditMode == GradientEditMode::PERCENTILE);
  bool isInteractiveMode = isCustomMode || isMinMaxMode || isWindowLevelMode || isPercentileMode;

  if (!isInteractiveMode) {
    return;
  }

  if (m_isDraggingPoint && m_currentPointIndex >= 0) {
    if (event->buttons() & Qt::LeftButton) {
      auto graph = m_customPlot->graph(0);
      double evx = m_customPlot->xAxis->pixelToCoord(event->pos().x());
      double evy = m_customPlot->yAxis->pixelToCoord(event->pos().y());

      // Handle threshold-based modes (MINMAX, Window/Level, Percentile) differently
      if (isMinMaxMode || isWindowLevelMode || isPercentileMode) {
        // In threshold-based modes, keep Y value constant and only allow horizontal movement
        double originalY = (graph->data()->begin() + m_currentPointIndex)->value;
        evy = originalY; // Keep Y constant

        static constexpr double OVERLAP_EPSILON = 0.001;
        // Apply additional constraints for min/max threshold points
        if (m_currentPointIndex == 1) {
          // This is the min threshold point - don't let it go past the max threshold point
          if (graph->data()->size() > 2) {
            double maxThresholdX = (graph->data()->begin() + 2)->key;
            evx = std::min(evx, maxThresholdX - OVERLAP_EPSILON); // Small epsilon to prevent overlap
          }
          // Also don't let it go before the first fixed point
          if (graph->data()->size() > 0) {
            double firstPointX = (graph->data()->begin())->key;
            evx = std::max(evx, firstPointX + OVERLAP_EPSILON);
          }
        } else if (m_currentPointIndex == 2) {
          // This is the max threshold point - don't let it go past the min threshold point
          if (graph->data()->size() > 1) {
            double minThresholdX = (graph->data()->begin() + 1)->key;
            evx = std::max(evx, minThresholdX + OVERLAP_EPSILON); // Small epsilon to prevent overlap
          }
          // Also don't let it go past the last fixed point
          if (graph->data()->size() > 3) {
            double lastPointX = (graph->data()->begin() + 3)->key;
            evx = std::min(evx, lastPointX - OVERLAP_EPSILON);
          }
        }
      }

      // see hoverpoints.cpp.
      // this will make sure we don't move past locked edges in the bounding rectangle.
      double px = evx, py = evy;
      bound_point(evx,
                  evy,
                  // we really want clipRect() here?  to capture zoomed region
                  QRectF(m_histogram._dataMin, 0.0f, m_histogram._dataMax - m_histogram._dataMin, 1.0f),
                  m_locks.at(m_currentPointIndex),
                  px,
                  py);

      // if we are dragging a point then move it
      (graph->data()->begin() + m_currentPointIndex)->value = py;
      (graph->data()->begin() + m_currentPointIndex)->key = px;

      // In MINMAX mode, don't sort - keep points in their fixed positions
      if (!isMinMaxMode) {
        // The point may have moved past other points, so sort,
        // and account for current point index possibly changing.
        // TODO should we always sort on every move? Or can we tell if we crossed another point?
        graph->data().data()->sort();
        // find new index of current point
        // find first point above x to know the index?
        int indexOfDataPoint = m_currentPointIndex;
        for (int n = 0; n < (graph->data()->size()); n++) {
          // get xy of each data pt in pixels. compare with scattersize.
          // first hit wins.
          double xn = (graph->data()->begin() + n)->key;
          if (px == xn) {
            // the index of x will be n.
            indexOfDataPoint = n;
            break;
          }
        }
        if (indexOfDataPoint != m_currentPointIndex) {
          m_currentPointIndex = indexOfDataPoint;
        }
      }

      // In threshold-based modes, emit interactivePointsChanged for real-time slider updates
      // Use the second and third points (indices 1 and 2) as the threshold points
      if ((isMinMaxMode || isWindowLevelMode || isPercentileMode) && graph->data()->size() >= 4) {
        double minThresholdX = (graph->data()->begin() + 1)->key;
        double maxThresholdX = (graph->data()->begin() + 2)->key;
        emit interactivePointsChanged(minThresholdX, maxThresholdX);
      }

      // emit( DataChanged() );

      emit gradientStopsChanged(this->buildStopsFromPlot());

      m_customPlot->replot();
    }
  }
}

QGradientStops
GradientEditor::buildStopsFromPlot()
{
  // build up coords from the customplot into the form of gradient stops
  QGradientStops stops;

  auto graph = m_customPlot->graph(0);
  for (int n = 0; n < (graph->data()->size()); n++) {
    auto dataIter = graph->data()->begin() + n;
    double x = dataIter->key;
    // skip duplicates?
    if (n + 1 < graph->data()->size() && x == graph->data()->at(n + 1)->key)
      continue;

    // rescale x to 0-1 range.
    x = (x - m_histogram._dataMin) / (m_histogram._dataMax - m_histogram._dataMin);
    double y = dataIter->value;

    QColor color = QColor::fromRgbF(y, y, y, y);
    if (x > 1.0) {
      LOG_ERROR << "control point x greater than 1";
      return stops;
    }

    stops << QGradientStop(x, color);
  }
  return stops;
}

void
GradientEditor::onPlotMouseRelease(QMouseEvent* event)
{
  Q_UNUSED(event);
  if (m_currentEditMode != GradientEditMode::CUSTOM && m_currentEditMode != GradientEditMode::MINMAX &&
      m_currentEditMode != GradientEditMode::WINDOW_LEVEL && m_currentEditMode != GradientEditMode::PERCENTILE) {
    return;
  }
  // if we were dragging a point then stop
  m_isDraggingPoint = false;
  m_currentPointIndex = -1;
  // re-enable axis dragging
  m_customPlot->axisRect()->setRangeDrag(Qt::Horizontal);
}

void
GradientEditor::onPlotMouseDoubleClick(QMouseEvent* event)
{
  // double click should reset zoom
  this->m_customPlot->rescaleAxes();
  this->m_customPlot->replot();
}

void
GradientEditor::onPlotMouseWheel(QWheelEvent* event)
{
  Q_UNUSED(event);
}

inline static bool
x_less_than(const QPointF& p1, const QPointF& p2)
{
  return p1.x() < p2.x();
}

inline static bool
controlpoint_x_less_than(const LutControlPoint& p1, const LutControlPoint& p2)
{
  return p1.first < p2.first;
}

QGradientStops
pointsToGradientStops(QPolygonF points)
{
  QGradientStops stops;
  std::sort(points.begin(), points.end(), x_less_than);

  for (int i = 0; i < points.size(); ++i) {
    qreal x = points.at(i).x();
    if (i + 1 < points.size() && x == points.at(i + 1).x())
      continue;
    float pixelvalue = points.at(i).y();
    // TODO future: let each point have a full RGBA color and use a color picker to assign it via dbl
    // click or some other means

    QColor color = QColor::fromRgbF(pixelvalue, pixelvalue, pixelvalue, pixelvalue);
    if (x > 1) {
      return stops;
    }

    stops << QGradientStop(x, color);
  }
  return stops;
}

void
GradientEditor::set_shade_points(const QPolygonF& points, QCustomPlot* plot, const Histogram& histogram)
{
  if (points.size() < 2) {
    return;
  }

  QGradientStops stops = pointsToGradientStops(points);

  m_locks.clear();
  if (points.size() > 0) {
    m_locks.resize(points.size());
    m_locks.fill(0);
  }
  m_locks[0] = GradientEditor::LockToLeft;
  m_locks[points.size() - 1] = GradientEditor::LockToRight;

  QVector<double> x, y;
  for (int i = 0; i < points.size(); ++i) {
    // incoming points x values are in 0-1 range which is normalized to histogram data range
    float dx = histogram._dataMin + points.at(i).x() * (histogram._dataMax - histogram._dataMin);
    x << dx;
    y << points.at(i).y();
  }
  plot->graph(0)->setData(x, y);
  plot->replot();
}

void
GradientEditor::setControlPoints(const std::vector<LutControlPoint>& points)
{
  QPolygonF pts_alpha;

  for (auto p : points) {
    pts_alpha << QPointF(p.first, p.second);
  }

  set_shade_points(pts_alpha, m_customPlot, m_histogram);
}

void
GradientEditor::wheelEvent(QWheelEvent* event)
{
  // wheel does nothing here!
  event->ignore();
}

GradientWidget::GradientWidget(const Histogram& histogram, GradientData* dataObject, QWidget* parent)
  : QWidget(parent)
  , m_histogram(histogram)
  , m_gradientData(dataObject)
{
  QVBoxLayout* mainGroupLayout = new QVBoxLayout(this);

  m_editor = new GradientEditor(m_histogram, this);
  mainGroupLayout->addWidget(m_editor);

  auto* sectionLayout = Controls::createAgaveFormLayout();

  QButtonGroup* btnGroup = new QButtonGroup(this);
  QPushButton* minMaxButton = new QPushButton("Min/Max");
  minMaxButton->setToolTip(tr("Min/Max"));
  minMaxButton->setStatusTip(tr("Choose Min/Max mode"));
  QPushButton* windowLevelButton = new QPushButton("Wnd/Lvl");
  windowLevelButton->setToolTip(tr("Window/Level"));
  windowLevelButton->setStatusTip(tr("Choose Window/Level mode"));
  QPushButton* isoButton = new QPushButton("Iso");
  isoButton->setToolTip(tr("Isovalue"));
  isoButton->setStatusTip(tr("Choose Isovalue mode"));
  QPushButton* pctButton = new QPushButton("Pct");
  pctButton->setToolTip(tr("Histogram Percentiles"));
  pctButton->setStatusTip(tr("Choose Histogram percentiles mode"));
  QPushButton* customButton = new QPushButton("Custom");
  customButton->setToolTip(tr("Custom"));
  customButton->setStatusTip(tr("Choose Custom editing mode"));

  static const int WINDOW_LEVEL_BTNID = 1;
  static const int ISO_BTNID = 2;
  static const int PCT_BTNID = 3;
  static const int CUSTOM_BTNID = 4;
  static const int MINMAX_BTNID = 5;
  static std::map<int, GradientEditMode> btnIdToGradientMode = { { WINDOW_LEVEL_BTNID, GradientEditMode::WINDOW_LEVEL },
                                                                 { ISO_BTNID, GradientEditMode::ISOVALUE },
                                                                 { PCT_BTNID, GradientEditMode::PERCENTILE },
                                                                 { MINMAX_BTNID, GradientEditMode::MINMAX },
                                                                 { CUSTOM_BTNID, GradientEditMode::CUSTOM } };
  static std::map<GradientEditMode, int> gradientModeToBtnId = { { GradientEditMode::WINDOW_LEVEL, WINDOW_LEVEL_BTNID },
                                                                 { GradientEditMode::ISOVALUE, ISO_BTNID },
                                                                 { GradientEditMode::PERCENTILE, PCT_BTNID },
                                                                 { GradientEditMode::MINMAX, MINMAX_BTNID },
                                                                 { GradientEditMode::CUSTOM, CUSTOM_BTNID } };
  static std::map<int, int> btnIdToStackedPage = {
    { WINDOW_LEVEL_BTNID, 1 }, { ISO_BTNID, 2 }, { PCT_BTNID, 3 }, { MINMAX_BTNID, 0 }, { CUSTOM_BTNID, 4 }
  };
  btnGroup->addButton(minMaxButton, MINMAX_BTNID);
  btnGroup->addButton(windowLevelButton, WINDOW_LEVEL_BTNID);
  btnGroup->addButton(isoButton, ISO_BTNID);
  btnGroup->addButton(pctButton, PCT_BTNID);
  btnGroup->addButton(customButton, CUSTOM_BTNID);
  QHBoxLayout* hbox = new QHBoxLayout();
  hbox->setSpacing(0);

  int initialButtonId = WINDOW_LEVEL_BTNID;
  GradientEditMode m = m_gradientData->m_activeMode;
  initialButtonId = gradientModeToBtnId[m];

  for (auto btn : btnGroup->buttons()) {
    btn->setCheckable(true);
    // set checked state initially.
    int btnid = btnGroup->id(btn);
    if (btnid == initialButtonId) {
      btn->setChecked(true);
    }
    hbox->addWidget(btn);
  }
  mainGroupLayout->addLayout(hbox);

  QWidget* firstPageWidget = new QWidget;
  auto* section0Layout = Controls::createAgaveFormLayout();
  firstPageWidget->setLayout(section0Layout);

  QWidget* secondPageWidget = new QWidget;
  auto* section1Layout = Controls::createAgaveFormLayout();
  secondPageWidget->setLayout(section1Layout);

  QWidget* thirdPageWidget = new QWidget;
  auto* section2Layout = Controls::createAgaveFormLayout();
  thirdPageWidget->setLayout(section2Layout);

  QWidget* fourthPageWidget = new QWidget;
  auto* section3Layout = Controls::createAgaveFormLayout();
  fourthPageWidget->setLayout(section3Layout);

  QWidget* fifthPageWidget = new QWidget;
  auto* section4Layout = Controls::createAgaveFormLayout();
  fifthPageWidget->setLayout(section4Layout);

  QStackedLayout* stackedLayout = new QStackedLayout(mainGroupLayout);
  stackedLayout->addWidget(firstPageWidget);
  stackedLayout->addWidget(secondPageWidget);
  stackedLayout->addWidget(thirdPageWidget);
  stackedLayout->addWidget(fourthPageWidget);
  stackedLayout->addWidget(fifthPageWidget);

  int initialStackedPageIndex = btnIdToStackedPage[initialButtonId];
  stackedLayout->setCurrentIndex(initialStackedPageIndex);
  // if this is not custom mode, then disable the gradient editor
  // m_editor->setEditable(m == GradientEditMode::CUSTOM);
  m_editor->setEditMode(m);

  connect(btnGroup,
          QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
          [this, btnGroup, stackedLayout](QAbstractButton* button) {
            int id = btnGroup->id(button);
            GradientEditMode modeToSet = btnIdToGradientMode[id];
            // if mode is not changing, we are done.
            if (modeToSet == this->m_gradientData->m_activeMode) {
              return;
            }
            this->m_gradientData->m_activeMode = modeToSet;

            stackedLayout->setCurrentIndex(btnIdToStackedPage[id]);

            // if this is not custom mode, then disable the gradient editor
            // m_editor->setEditable(modeToSet == GradientEditMode::CUSTOM);
            m_editor->setEditMode(modeToSet);

            this->forceDataUpdate();
          });

  minu16Slider = new QIntSlider();
  minu16Slider->setStatusTip(tr("Minimum u16 value"));
  minu16Slider->setToolTip(tr("Set minimum u16 value"));
  minu16Slider->setRange(m_histogram._dataMin, m_histogram._dataMax);
  minu16Slider->setSingleStep(1);
  minu16Slider->setValue(m_gradientData->m_minu16);
  section0Layout->addRow("Min u16", minu16Slider);
  maxu16Slider = new QIntSlider();
  maxu16Slider->setStatusTip(tr("Maximum u16 value"));
  maxu16Slider->setToolTip(tr("Set maximum u16 value"));
  maxu16Slider->setRange(m_histogram._dataMin, m_histogram._dataMax);
  maxu16Slider->setSingleStep(1);
  maxu16Slider->setValue(m_gradientData->m_maxu16);
  section0Layout->addRow("Max u16", maxu16Slider);
  connect(minu16Slider, &QIntSlider::valueChanged, [this](int i) {
    this->m_gradientData->m_minu16 = i;
    this->onSetMinMax(i, this->m_gradientData->m_maxu16);
  });
  connect(maxu16Slider, &QIntSlider::valueChanged, [this](int i) {
    this->m_gradientData->m_maxu16 = i;
    this->onSetMinMax(this->m_gradientData->m_minu16, i);
  });

  windowSlider = new QNumericSlider();
  windowSlider->setStatusTip(tr("Window"));
  windowSlider->setToolTip(tr("Set size of range of intensities"));
  windowSlider->setRange(0.0, 1.0);
  windowSlider->setSingleStep(0.01);
  windowSlider->setDecimals(3);
  windowSlider->setValue(m_gradientData->m_window);
  section1Layout->addRow("Window", windowSlider);
  levelSlider = new QNumericSlider();
  levelSlider->setStatusTip(tr("Level"));
  levelSlider->setToolTip(tr("Set level of mid intensity"));
  levelSlider->setRange(0.0, 1.0);
  levelSlider->setSingleStep(0.01);
  levelSlider->setDecimals(3);
  levelSlider->setValue(m_gradientData->m_level);
  section1Layout->addRow("Level", levelSlider);
  connect(windowSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_window = d;
    this->onSetWindowLevel(d, levelSlider->value());
  });
  connect(levelSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_level = d;
    this->onSetWindowLevel(windowSlider->value(), d);
  });

  isovalueSlider = new QNumericSlider();
  isovalueSlider->setStatusTip(tr("Isovalue"));
  isovalueSlider->setToolTip(tr("Set Isovalue"));
  isovalueSlider->setRange(0.0, 1.0);
  isovalueSlider->setSingleStep(0.01);
  isovalueSlider->setDecimals(3);
  isovalueSlider->setValue(m_gradientData->m_isovalue);
  section2Layout->addRow("Isovalue", isovalueSlider);
  isorangeSlider = new QNumericSlider();
  isorangeSlider->setStatusTip(tr("Isovalue range"));
  isorangeSlider->setToolTip(tr("Set range above and below isovalue"));
  isorangeSlider->setRange(0.0, 1.0);
  isorangeSlider->setSingleStep(0.01);
  isorangeSlider->setDecimals(3);
  isorangeSlider->setValue(m_gradientData->m_isorange);
  section2Layout->addRow("Iso-range", isorangeSlider);
  connect(isovalueSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_isovalue = d;
    this->onSetIsovalue(d, isorangeSlider->value());
  });
  connect(isorangeSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_isorange = d;
    this->onSetIsovalue(isovalueSlider->value(), d);
  });

  pctLowSlider = new QNumericSlider();
  pctLowSlider->setStatusTip(tr("Low percentile"));
  pctLowSlider->setToolTip(tr("Set bottom percentile"));
  pctLowSlider->setRange(0.0, 1.0);
  pctLowSlider->setSingleStep(0.01);
  pctLowSlider->setDecimals(3);
  pctLowSlider->setValue(m_gradientData->m_pctLow);
  section3Layout->addRow("Pct Min", pctLowSlider);
  pctHighSlider = new QNumericSlider();
  pctHighSlider->setStatusTip(tr("High percentile"));
  pctHighSlider->setToolTip(tr("Set top percentile"));
  pctHighSlider->setRange(0.0, 1.0);
  pctHighSlider->setSingleStep(0.01);
  pctHighSlider->setDecimals(3);
  pctHighSlider->setValue(m_gradientData->m_pctHigh);
  section3Layout->addRow("Pct Max", pctHighSlider);
  connect(pctLowSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_pctLow = d;
    this->onSetHistogramPercentiles(d, pctHighSlider->value());
  });
  connect(pctHighSlider, &QNumericSlider::valueChanged, [this](double d) {
    this->m_gradientData->m_pctHigh = d;
    this->onSetHistogramPercentiles(pctLowSlider->value(), d);
  });

  mainGroupLayout->addLayout(sectionLayout);
  mainGroupLayout->addStretch(1);

  connect(m_editor, &GradientEditor::gradientStopsChanged, this, &GradientWidget::onGradientStopsChanged);
  connect(m_editor, &GradientEditor::interactivePointsChanged, this, &GradientWidget::onInteractivePointsChanged);

  forceDataUpdate();
}

void
GradientWidget::forceDataUpdate()
{
  GradientEditMode mode = this->m_gradientData->m_activeMode;

  switch (mode) {
    case GradientEditMode::WINDOW_LEVEL:
      this->onSetWindowLevel(this->m_gradientData->m_window, this->m_gradientData->m_level);
      break;
    case GradientEditMode::ISOVALUE:
      this->onSetIsovalue(this->m_gradientData->m_isovalue, this->m_gradientData->m_isorange);
      break;
    case GradientEditMode::PERCENTILE:
      this->onSetHistogramPercentiles(this->m_gradientData->m_pctLow, this->m_gradientData->m_pctHigh);
      break;
    case GradientEditMode::MINMAX:
      this->onSetMinMax(this->m_gradientData->m_minu16, this->m_gradientData->m_maxu16);
      break;
    case GradientEditMode::CUSTOM: {
      m_editor->setControlPoints(this->m_gradientData->m_customControlPoints);
      QGradientStops stops = vectorToGradientStops(this->m_gradientData->m_customControlPoints);
      emit gradientStopsChanged(stops);
    } break;
    default:
      LOG_ERROR << "Bad gradient editor mode";
      break;
  }
}

void
GradientWidget::onGradientStopsChanged(const QGradientStops& stops)
{
  // update the data stored in m_gradientData
  // depending on our mode:
  if (m_gradientData->m_activeMode == GradientEditMode::CUSTOM) {
    m_gradientData->m_customControlPoints.clear();
    for (int i = 0; i < stops.size(); ++i) {
      m_gradientData->m_customControlPoints.push_back(LutControlPoint(stops.at(i).first, stops.at(i).second.alphaF()));
    }
    emit gradientStopsChanged(stops);
  } else if (m_gradientData->m_activeMode == GradientEditMode::WINDOW_LEVEL) {
    // extract window and level from the stops - use second and third points (threshold points)
    std::vector<LutControlPoint> points = gradientStopsToVector(const_cast<QGradientStops&>(stops));
    if (points.size() < 4) {
      return;
    }
    std::sort(points.begin(), points.end(), controlpoint_x_less_than);
    float low = points[1].first;  // Second point (low threshold)
    float high = points[2].first; // Third point (high threshold)
    float window = high - low;
    float level = (high + low) * 0.5f;
    m_gradientData->m_window = window;
    m_gradientData->m_level = level;

    // update the sliders to match:
    windowSlider->setValue(window);
    levelSlider->setValue(level);

  } else if (m_gradientData->m_activeMode == GradientEditMode::ISOVALUE) {
    // extract isovalue and range from the stops
    std::vector<LutControlPoint> points = gradientStopsToVector(const_cast<QGradientStops&>(stops));
    if (points.size() < 2) {
      return;
    }
    std::sort(points.begin(), points.end(), controlpoint_x_less_than);
    float low = points[0].first;
    float high = points[1].first;
    float isovalue = (high + low) * 0.5f;
    float isorange = high - low;
    m_gradientData->m_isovalue = isovalue;
    m_gradientData->m_isorange = isorange;

    // update the sliders to match:
    isovalueSlider->setValue(isovalue);
    isorangeSlider->setValue(isorange);
  } else if (m_gradientData->m_activeMode == GradientEditMode::PERCENTILE) {
    // get percentiles from the stops and histogram - use second and third points (threshold points)
    std::vector<LutControlPoint> points = gradientStopsToVector(const_cast<QGradientStops&>(stops));
    if (points.size() < 4) {
      return;
    }
    std::sort(points.begin(), points.end(), controlpoint_x_less_than);
    float low = points[1].first;  // Second point (low threshold)
    float high = points[2].first; // Third point (high threshold)
    // calculate percentiles from the histogram:
    uint16_t ulow = m_histogram._dataMin + static_cast<uint16_t>(low * (m_histogram._dataMax - m_histogram._dataMin));
    uint16_t uhigh = m_histogram._dataMin + static_cast<uint16_t>(high * (m_histogram._dataMax - m_histogram._dataMin));
    float pctLow = 0.0f, pctHigh = 1.0f;
    m_histogram.computePercentile(ulow, pctLow);
    m_histogram.computePercentile(uhigh, pctHigh);
    m_gradientData->m_pctLow = pctLow;
    m_gradientData->m_pctHigh = pctHigh;

    // update the sliders to match:
    pctLowSlider->setValue(pctLow);
    pctHighSlider->setValue(pctHigh);
  } else if (m_gradientData->m_activeMode == GradientEditMode::MINMAX) {
    // get absolute min/max from the stops - use second and third points (threshold points)
    std::vector<LutControlPoint> points = gradientStopsToVector(const_cast<QGradientStops&>(stops));
    if (points.size() < 4) {
      return;
    }
    std::sort(points.begin(), points.end(), controlpoint_x_less_than);
    // turn the second and third points' x values into u16 intensities from the histogram range:
    float low = points[1].first;  // Second point (min threshold)
    float high = points[2].first; // Third point (max threshold)
    // calculate percentiles from the histogram:
    uint16_t ulow = m_histogram._dataMin + static_cast<uint16_t>(low * (m_histogram._dataMax - m_histogram._dataMin));
    uint16_t uhigh = m_histogram._dataMin + static_cast<uint16_t>(high * (m_histogram._dataMax - m_histogram._dataMin));
    m_gradientData->m_minu16 = ulow;
    m_gradientData->m_maxu16 = uhigh;

    // update the sliders to match:
    minu16Slider->setValue(ulow);
    maxu16Slider->setValue(uhigh);
  }
}

void
GradientWidget::onSetHistogramPercentiles(float pctLow, float pctHigh)
{
  float window, level;
  m_histogram.computeWindowLevelFromPercentiles(pctLow, pctHigh, window, level);
  this->onSetWindowLevel(window, level);
}

void
GradientWidget::onSetWindowLevel(float window, float level)
{
  std::vector<LutControlPoint> points;
  static const float epsilon = 0.000001f;
  window = std::max(window, epsilon);
  float lowEnd = level - window * 0.5f;
  float highEnd = level + window * 0.5f;
  if (lowEnd <= 0.0f) {
    float val = -lowEnd / (highEnd - lowEnd);
    points.push_back({ 0.0f, val });
  } else {
    points.push_back({ 0.0f, 0.0f });
    points.push_back({ lowEnd, 0.0f });
  }
  if (highEnd >= 1.0f) {
    float val = (1.0f - lowEnd) / (highEnd - lowEnd);
    points.push_back({ 1.0f, val });
  } else {
    points.push_back({ highEnd, 1.0f });
    points.push_back({ 1.0f, 1.0f });
  }
  m_editor->setControlPoints(points);
  emit gradientStopsChanged(vectorToGradientStops(points));
}

void
GradientWidget::onSetMinMax(uint16_t minu16, uint16_t maxu16)
{
  // these need to be relative to the data range of the channel, not absolute!
  float relativeMin = normalizeInt(minu16, m_histogram._dataMin, m_histogram._dataMax);
  float relativeMax = normalizeInt(maxu16, m_histogram._dataMin, m_histogram._dataMax);
  relativeMin = std::max(relativeMin, 0.0f);
  relativeMax = std::min(relativeMax, 1.0f);
  if (relativeMin >= relativeMax) {
    LOG_ERROR << "Min value is greater than or equal to max value: " << minu16 << " >= " << maxu16
              << ", datarange=" << m_histogram.dataRange();
    return;
  }
  float window = relativeMax - relativeMin;
  float level = (relativeMax + relativeMin) / 2.0f;
  this->onSetWindowLevel(window, level);
}

void
GradientWidget::onSetIsovalue(float isovalue, float width)
{
  std::vector<LutControlPoint> points;
  float lowEnd = isovalue - width * 0.5f;
  float highEnd = isovalue + width * 0.5f;
  static const float epsilon = 0.00001f;
  points.push_back({ 0.0f, 0.0f });
  points.push_back({ lowEnd - epsilon, 0.0f });
  points.push_back({ lowEnd + epsilon, 1.0f });
  points.push_back({ highEnd - epsilon, 1.0f });
  points.push_back({ highEnd + epsilon, 0.0f });
  points.push_back({ 1.0f, 0.0f });
  m_editor->setControlPoints(points);
  emit gradientStopsChanged(vectorToGradientStops(points));
}

void
GradientWidget::onInteractivePointsChanged(float minIntensity, float maxIntensity)
{
  // Handle different modes appropriately
  if (m_gradientData->m_activeMode == GradientEditMode::MINMAX) {
    // Convert from graph coordinates (histogram data range) to u16 intensity values
    uint16_t minu16 = static_cast<uint16_t>(minIntensity);
    uint16_t maxu16 = static_cast<uint16_t>(maxIntensity);

    // Ensure values are within valid range
    minu16 = std::max(minu16, static_cast<uint16_t>(m_histogram._dataMin));
    maxu16 = std::min(maxu16, static_cast<uint16_t>(m_histogram._dataMax));

    // Update the data
    m_gradientData->m_minu16 = minu16;
    m_gradientData->m_maxu16 = maxu16;

    // Update sliders without triggering their signals (to avoid feedback loops)
    if (minu16Slider) {
      minu16Slider->blockSignals(true);
      minu16Slider->setValue(minu16);
      minu16Slider->blockSignals(false);
    }
    if (maxu16Slider) {
      maxu16Slider->blockSignals(true);
      maxu16Slider->setValue(maxu16);
      maxu16Slider->blockSignals(false);
    }
  } else if (m_gradientData->m_activeMode == GradientEditMode::WINDOW_LEVEL) {
    // Convert intensities to normalized values (0-1 range)
    uint16_t minInt = static_cast<uint16_t>(minIntensity);
    uint16_t maxInt = static_cast<uint16_t>(maxIntensity);
    float relativeMin = normalizeInt<uint16_t>(minInt, m_histogram._dataMin, m_histogram._dataMax);
    float relativeMax = normalizeInt<uint16_t>(maxInt, m_histogram._dataMin, m_histogram._dataMax);

    // Calculate window and level from the threshold points
    float window = relativeMax - relativeMin;
    float level = (relativeMax + relativeMin) * 0.5f;

    // Update the data
    m_gradientData->m_window = window;
    m_gradientData->m_level = level;

    // Update sliders without triggering their signals
    if (windowSlider) {
      windowSlider->blockSignals(true);
      windowSlider->setValue(window);
      windowSlider->blockSignals(false);
    }
    if (levelSlider) {
      levelSlider->blockSignals(true);
      levelSlider->setValue(level);
      levelSlider->blockSignals(false);
    }
  } else if (m_gradientData->m_activeMode == GradientEditMode::PERCENTILE) {
    // Convert intensities to u16 values first
    uint16_t minu16 = static_cast<uint16_t>(std::max(minIntensity, (float)m_histogram._dataMin));
    uint16_t maxu16 = static_cast<uint16_t>(std::min(maxIntensity, (float)m_histogram._dataMax));

    // Calculate percentiles by converting intensity to bin index and using cumulative counts
    float pctLow = 0.0f, pctHigh = 1.0f;

    if (m_histogram._pixelCount > 0 && !m_histogram._ccounts.empty()) {
      // For low percentile
      if (minu16 <= m_histogram._dataMin) {
        pctLow = 0.0f;
      } else if (minu16 >= m_histogram._dataMax) {
        pctLow = 1.0f;
      } else {
        m_histogram.computePercentile(minu16, pctLow);
      }

      // For high percentile
      if (maxu16 <= m_histogram._dataMin) {
        pctHigh = 0.0f;
      } else if (maxu16 >= m_histogram._dataMax) {
        pctHigh = 1.0f;
      } else {
        m_histogram.computePercentile(maxu16, pctHigh);
      }
    }

    // Update the data
    m_gradientData->m_pctLow = pctLow;
    m_gradientData->m_pctHigh = pctHigh;

    // Update sliders without triggering their signals
    if (pctLowSlider) {
      pctLowSlider->blockSignals(true);
      pctLowSlider->setValue(pctLow);
      pctLowSlider->blockSignals(false);
    }
    if (pctHighSlider) {
      pctHighSlider->blockSignals(true);
      pctHighSlider->setValue(pctHigh);
      pctHighSlider->blockSignals(false);
    }
  }
}
