#include "ViewToolbar.h"

#include <QFile>
#include <QHBoxLayout>
#include <QIcon>
#include <QIconEngine>
#include <QPainter>
#include <QPixmap>
#include <QPixmapCache>
#include <QPushButton>
#include <QSvgRenderer>

#if 0
class PaletteIconEngine : public QIconEngine
{
public:
  PaletteIconEngine();
  PaletteIconEngine(const PaletteIconEngine& other);
  ~PaletteIconEngine();

  void paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State state) override;
  QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override;

  void addFile(const QString& fileName, const QSize& size, QIcon::Mode mode, QIcon::State state) override;

  QString key() const override;
  QIconEngine* clone() const override;

  QList<QSize> availableSizes(QIcon::Mode mode, QIcon::State state) override;

  void virtual_hook(int id, void* data) override;

private:
  QScopedPointer<QSvgRenderer> renderer_;
  QString src_file_;
};

static QString
actualFilename(const QString& filename)
{
  if (QFile::exists(filename))
    return filename;

  QString fn = filename.mid(0, filename.lastIndexOf('.'));
  if (QFile::exists(fn))
    return fn;

  return fn + ".svg";
}

static QColor
getIconColor(QIcon::Mode mode, QIcon::State state)
{
  Q_UNUSED(state);
  QPalette::ColorGroup color_group = QPalette::Active;
  if (mode == QIcon::Disabled)
    color_group = QPalette::Disabled;
  return QPalette().color(color_group, QPalette::WindowText);
}

static QPixmap
renderIcon(QSvgRenderer* renderer, const QSize& size, const QBrush& brush)
{
  QPixmap output(size);
  output.fill(Qt::transparent);

  QPainter p(&output);
  renderer->render(&p);

  p.setCompositionMode(QPainter::CompositionMode_SourceIn);

  p.setPen(Qt::NoPen);
  p.setBrush(brush);

  p.drawRect(output.rect());

  return output;
}

PaletteIconEngine::PaletteIconEngine()
{
  renderer_.reset(new QSvgRenderer());
}

PaletteIconEngine::PaletteIconEngine(const PaletteIconEngine& other)
  : QIconEngine(other)
{
  src_file_ = other.src_file_;
  renderer_.reset(new QSvgRenderer());
  if (other.renderer_->isValid())
    renderer_->load(other.src_file_);
}

PaletteIconEngine::~PaletteIconEngine() = default;

void
PaletteIconEngine::paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State state)
{
  // "direct rendereng" using given painter is not possible
  // because colorization logic modifies already painted area
  // such behavior is not acceptable, so render icon to pixmap first
  QSize size = rect.size() * painter->device()->devicePixelRatioF();
  QPixmap pxm = pixmap(size, mode, state);
  // set device pixel ratio exactly before painting,
  // this will allow to reuse the same cached pixmap
  pxm.setDevicePixelRatio(painter->device()->devicePixelRatioF());
  painter->drawPixmap(rect, pxm);
}

QPixmap
PaletteIconEngine::pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state)
{
  // unfortunately, default implementation (call paint() on newly created QPixmap)
  // doesn't initialize (fill) QPixmap with transparent color, so artifacts may happen
  // so, it is better to implement pixmap() function and use it in paint() implementation
  QColor color = getIconColor(mode, state);
  QString pmckey =
    QString("pie_%1:%2x%3:%4").arg(src_file_).arg(size.width()).arg(size.height()).arg(color.name(QColor::HexArgb));
  QPixmap pxm;
  if (!QPixmapCache::find(pmckey, &pxm)) {
    pxm = renderIcon(renderer_.data(), size, color);
    QPixmapCache::insert(pmckey, pxm);
  }
  return pxm;
}

void
PaletteIconEngine::addFile(const QString& fileName, const QSize& size, QIcon::Mode mode, QIcon::State state)
{
  Q_UNUSED(size);
  Q_UNUSED(mode);
  Q_UNUSED(state);
  QString filename = actualFilename(fileName);
  if (filename == src_file_)
    return;
  if (renderer_->load(filename))
    src_file_ = filename;
}

QString
PaletteIconEngine::key() const
{
  return QLatin1String("palette");
}

QIconEngine*
PaletteIconEngine::clone() const
{
  return new PaletteIconEngine(*this);
}

QList<QSize>
PaletteIconEngine::availableSizes(QIcon::Mode mode, QIcon::State state)
{
  Q_UNUSED(mode);
  Q_UNUSED(state);
  QList<QSize> sizes;
  sizes << QSize(512, 512); // just workaround to make tray icon visible on KDE
  return sizes;
}

void
PaletteIconEngine::virtual_hook(int id, void* data)
{
  switch (id) {
    case QIconEngine::IsNullHook:
      *reinterpret_cast<bool*>(data) = (!renderer_ || !renderer_->isValid());
      break;

    default:
      QIconEngine::virtual_hook(id, data);
  }
}

class SwitchingIcon : public QIcon
{
  SwitchingIcon(const QString& file_name)
    : QIcon()
  {
    // QString f1 = QString("switch1/%1").arg(file_name);
    // QString f2 = QString("switch2/%1").arg(file_name);
    QIconEngine* engine = new PaletteIconEngine();
    engine->addFile(file_name, QSize(512, 512), QIcon::Normal, QIcon::Off);
    QIcon(engine);
  }
};
#endif

ViewToolbar::ViewToolbar(QWidget* parent)
  : QWidget(parent)
{
  QHBoxLayout* toolbarLayout = new QHBoxLayout(this);
  toolbarLayout->setSpacing(0);
  toolbarLayout->setContentsMargins(0, 0, 0, 0);

  frameViewButton = new QPushButton(QIcon(":/icons/frameView.svg"), "", this);
  frameViewButton->setToolTip(QString("<FONT>Frame view</FONT>"));
  frameViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frameViewButton->adjustSize();
  frameViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frameViewButton);

  orthoViewButton = new QPushButton(QIcon(":/icons/orthoView.svg"), "", this);
  orthoViewButton->setToolTip(QString("<FONT>Ortho/Persp view</FONT>"));
  orthoViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  orthoViewButton->adjustSize();
  orthoViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(orthoViewButton);

  topViewButton = new QPushButton(QIcon(":/icons/topView.svg"), "", this);
  topViewButton->setToolTip(QString("<FONT>Top view (-Y)</FONT>"));
  topViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  topViewButton->adjustSize();
  topViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(topViewButton);

  bottomViewButton = new QPushButton(QIcon(":/icons/bottomView.svg"), "", this);
  bottomViewButton->setToolTip(QString("<FONT>Bottom view (+Y)</FONT>"));
  bottomViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  bottomViewButton->adjustSize();
  bottomViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(bottomViewButton);

  frontViewButton = new QPushButton(QIcon(":/icons/frontView.svg"), "", this);
  frontViewButton->setToolTip(QString("<FONT>Front view (-Z)</FONT>"));
  frontViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  frontViewButton->adjustSize();
  frontViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(frontViewButton);

  backViewButton = new QPushButton(QIcon(":/icons/backView.svg"), "", this);
  backViewButton->setToolTip(QString("<FONT>Back view (+Z)</FONT>"));
  backViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  backViewButton->adjustSize();
  backViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(backViewButton);

  leftViewButton = new QPushButton(QIcon(":/icons/leftView.svg"), "", this);
  leftViewButton->setToolTip(QString("<FONT>Left view (+X)</FONT>"));
  leftViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  leftViewButton->adjustSize();
  leftViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(leftViewButton);

  rightViewButton = new QPushButton(QIcon(":/icons/rightView.svg"), "", this);
  rightViewButton->setToolTip(QString("<FONT>Right view (-X)</FONT>"));
  rightViewButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
  rightViewButton->adjustSize();
  rightViewButton->setFocusPolicy(Qt::NoFocus);
  toolbarLayout->addWidget(rightViewButton);

  //   int width = zoomFitButton->fontMetrics().boundingRect(zoomFitButton->text()).width() * 2 + 7;
  //   zoomInButton->setMaximumWidth(width);
  //   zoomOutButton->setMaximumWidth(width);
  //   zoomFitButton->setMaximumWidth(width);

  toolbarLayout->addStretch();

  setLayout(toolbarLayout);
  setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  // move(0, 0);
}

ViewToolbar::~ViewToolbar() {}

void
ViewToolbar::positionToolbar()
{
  // float over parent or lay out in a vboxlayout?
  // auto s = parent()->size();
  // move(s.width() - width() - TOOLBAR_INSET, s.height() - height() - TOOLBAR_INSET);
  // move(0, 0);
}
