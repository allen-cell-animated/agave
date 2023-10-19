#pragma once

#include <QDockWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QPointer>
#include <QPushButton>
#include <QSize>
#include <QWidget>

/**
 * A QDockWidget which provides the possibility to collapse the dock.
 *
 * @see QDockWidget
 */
class CollapsibleDockWidget : public QDockWidget
{
  Q_OBJECT

public:
  /**
   * Creates a new CollapsibleDockWidget.
   * This calls the constructor of the QDockWidget.
   *
   * @param title The window title of the DockWidget.
   * @param parent The parent QWidget.
   * @param The flags to create the DockWidget with.
   *
   * @see QDockWidget
   */
  explicit CollapsibleDockWidget(const QString& title, QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());

  /**
   * Creates a new CollapsibleDockWidget.
   * This calls the constructor of the QDockWidget.
   *
   * @param parent The parent QWidget.
   * @param The flags to create the DockWidget with.
   *
   * @see QDockWidget
   */
  explicit CollapsibleDockWidget(QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());

  /**
   * Sets a QWidget to show in the DockWidget.
   * If you use this method the DockWidget is collapsible.
   *
   * @param w The widget to set.
   */
  void setCollapsibleWidget(QWidget* w);

  /**
   * Returns is the widget is collapsed.
   *
   * @return The current state of the widdget.
   */
  bool isCollapsed();

  /**
   * A method to inform the widget, that the window title has been changed.
   */
  void windowTitleChanged();

public slots:
  /**
   * Sets the collapsed state. The widget will be collapsed if and only if "collapsed" is true.
   *
   * @param collapsed The new state of the widget.
   */
  void setCollapsed(bool collapsed);

  /**
   * Toggles the collapsed state of the widget.
   */
  void toggleCollapsed();

private:
  class InnerWidgetWrapper : public QWidget
  {
  public:
    InnerWidgetWrapper(QDockWidget* parent);

    void setWidget(QWidget* widget);

    bool isCollapsed();

    void setCollapsed(bool collapsed);

    QSize const& getOldMaximumSizeParent() const;

  private:
    QPointer<QWidget> widget;
    QPointer<QHBoxLayout> hlayout;
    int widget_height;
    QSize oldSize;
    QSize oldMinimumSizeParent;
    QSize oldMaximumSizeParent;
    QSize oldMinimumSize;
    QSize oldMaximumSize;
  };

  class TitleBar : public QWidget
  {
  public:
    TitleBar(QWidget* parent);

    void windowTitleChanged();

    void showTitle(bool show);
    void setCollapsed(bool collapsed);

  private:
    QPointer<QHBoxLayout> hlayout;
    QPointer<QPushButton> collapse;
    QPointer<QPushButton> close;
    QPointer<QPushButton> undock;
    QPointer<QLabel> title;
  };

private slots:
  void setCollapsedSizes();
};
