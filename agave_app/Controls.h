#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QEvent>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QInputDialog>
#include <QLineEdit>
#include <QListView>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStyledItemDelegate>

class QColorPushButton : public QPushButton
{
  Q_OBJECT

public:
  QColorPushButton(QWidget* pParent = NULL);

  virtual void paintEvent(QPaintEvent* pPaintEvent);
  virtual void mousePressEvent(QMouseEvent* pEvent);

  int GetMargin(void) const;
  void SetMargin(const int& Margin);
  int GetRadius(void) const;
  void SetRadius(const int& Radius);
  QColor GetColor(void) const;
  void SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
  void OnCurrentColorChanged(const QColor& Color);

signals:
  void currentColorChanged(const QColor&);

private:
  int m_Margin;
  int m_Radius;
  QColor m_Color;
};

class QColorSelector : public QFrame
{
  Q_OBJECT

public:
  QColorSelector(QWidget* pParent = NULL);

  //	virtual QSize sizeHint() const;

  QColor GetColor(void) const;
  void SetColor(const QColor& Color, bool BlockSignals = false);

private slots:
  void OnCurrentColorChanged(const QColor& Color);

signals:
  void currentColorChanged(const QColor&);

private:
  QGridLayout m_MainLayout;
  QColorPushButton m_ColorButton;
  QComboBox m_ColorCombo;
};

class QDoubleSlider : public QSlider
{
  Q_OBJECT

public:
  QDoubleSlider(QWidget* pParent = NULL);

  void setRange(double Min, double Max);
  void setMinimum(double Min);
  double minimum() const;
  void setMaximum(double Max);
  double maximum() const;
  double value() const;

  double sliderPositionToValue(int pos) const;

public slots:
  void setValue(int value);
  void setValue(double Value, bool BlockSignals = false);

private slots:

signals:
  void valueChanged(double Value);
  void rangeChanged(double Min, double Max);

private:
  double m_Multiplier;
};

class QDoubleSpinner : public QDoubleSpinBox
{
  Q_OBJECT

public:
  QDoubleSpinner(QWidget* pParent = NULL);

  virtual QSize sizeHint() const;
  void setValue(double Value, bool BlockSignals = false);
};

class QNumericSlider : public QWidget
{
  Q_OBJECT
public:
  QNumericSlider(QWidget* pParent = NULL);

  double value(void) const;
  void setValue(double value, bool BlockSignals = false);
  void setRange(double rmin, double rmax);
  void setSingleStep(double val);
  void setDecimals(int decimals);
  void setSuffix(const QString&);

  void setTracking(bool enable);

private slots:
  void OnValueChanged(double value);

signals:
  void valueChanged(double value);

private:
  QGridLayout m_layout;
  QDoubleSpinner m_spinner;
  QDoubleSlider m_slider;
};

class QIntSlider : public QWidget
{
  Q_OBJECT
public:
  QIntSlider(QWidget* pParent = NULL);

  int value(void) const;
  int maximum(void) const;
  void setValue(int value, bool BlockSignals = false);
  void setRange(int rmin, int rmax);
  void setSingleStep(int val);
  void setSuffix(const QString&);

  void setTickPosition(QSlider::TickPosition position);
  void setTickInterval(int ti);

  void setTracking(bool enable);
private slots:
  void OnValueChanged(int value);

signals:
  void valueChanged(int value);

private:
  QGridLayout m_layout;
  QSpinBox m_spinner;
  QSlider m_slider;
};

class Controls
{
public:
  static QFormLayout* createFormLayout(QWidget* parent = nullptr)
  {
    QFormLayout* layout = new QFormLayout(parent);
    initFormLayout(*layout);
    return layout;
  }
  static void initFormLayout(QFormLayout& layout)
  {
    layout.setRowWrapPolicy(QFormLayout::DontWrapRows);
    layout.setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    layout.setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    layout.setLabelAlignment(Qt::AlignLeft);
  }
};

/**
 * @brief QComboBox with support of checkboxes
 * http://stackoverflow.com/questions/8422760/combobox-of-checkboxes
 */
class QCheckList : public QComboBox
{
  Q_OBJECT

public:
  /**
   * @brief Additional value to Qt::CheckState when some checkboxes are Qt::PartiallyChecked
   */
  static const int StateUnknown = 3;

private:
  QStandardItemModel* m_model;
  QString m_titleText;

signals:
  void globalCheckStateChanged(int);

public:
  QCheckList(QWidget* _parent = 0)
    : QComboBox(_parent)
  {
    m_model = new QStandardItemModel();
    setModel(m_model);

    setEditable(true);
    lineEdit()->setReadOnly(true);
    lineEdit()->installEventFilter(this);
    setItemDelegate(new QCheckListStyledItemDelegate(this));

    connect(lineEdit(), &QLineEdit::selectionChanged, lineEdit(), &QLineEdit::deselect);
    connect((QListView*)view(), SIGNAL(pressed(QModelIndex)), this, SLOT(on_itemPressed(QModelIndex)));
    connect(m_model, SIGNAL(dataChanged(QModelIndex, QModelIndex, QVector<int>)), this, SLOT(on_modelDataChanged()));
  }

  QCheckList(const QString& title, QWidget* _parent = 0) 
      : QCheckList(_parent)
  {
    m_titleText = title;
  }

  ~QCheckList() { delete m_model; }

  void setTitleText(const QString& text)
  {
	m_titleText = text;
	updateText();
  }

  /**
   * @brief Adds a item to the checklist (setChecklist must have been called)
   * @return the new QStandardItem
   */
  QStandardItem* addCheckItem(const QString& label, const QVariant& data, const Qt::CheckState checkState)
  {
    QStandardItem* item = new QStandardItem(label);
    item->setCheckState(checkState);
    item->setData(data);
    item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);

    m_model->appendRow(item);

    updateText();

    return item;
  }

  /**
   * @brief Computes the global state of the checklist :
   *      - if there is no item: StateUnknown
   *      - if there is at least one item partially checked: StateUnknown
   *      - if all items are checked: Qt::Checked
   *      - if no item is checked: Qt::Unchecked
   *      - else: Qt::PartiallyChecked
   */
  int globalCheckState()
  {
    int nbRows = m_model->rowCount(), nbChecked = 0, nbUnchecked = 0;

    if (nbRows == 0) {
      return StateUnknown;
    }

    for (int i = 0; i < nbRows; i++) {
      if (m_model->item(i)->checkState() == Qt::Checked) {
        nbChecked++;
      } else if (m_model->item(i)->checkState() == Qt::Unchecked) {
        nbUnchecked++;
      } else {
        return StateUnknown;
      }
    }

    return nbChecked == nbRows ? Qt::Checked : nbUnchecked == nbRows ? Qt::Unchecked : Qt::PartiallyChecked;
  }

protected:
  bool eventFilter(QObject* _object, QEvent* _event)
  {
    if (_object == lineEdit() && _event->type() == QEvent::MouseButtonPress) {
      showPopup();
      return true;
    }

    return false;
  }

private:
  void updateText()
  {
    QString text = m_titleText;
    int numChecked = 0;
    switch (globalCheckState()) {
      case Qt::Checked:
        //text = m_allCheckedText;
        text += QString(" (%1/%2 selected)").arg(m_model->rowCount()).arg(m_model->rowCount());
        break;

      case Qt::Unchecked:
        //text = m_noneCheckedText;
        text += QString(" (0/%1 selected)").arg(m_model->rowCount());
        break;

      case Qt::PartiallyChecked:

        for (int i = 0; i < m_model->rowCount(); i++) {
          if (m_model->item(i)->checkState() == Qt::Checked) {
            numChecked++;
            //if (!text.isEmpty()) {
            //  text += ", ";
            //}
            //text += m_model->item(i)->text();
          }
        }
        text += QString(" (%1/%2 selected)").arg(numChecked).arg(m_model->rowCount());
        break;

      default:
        text = m_titleText;
    }

    lineEdit()->setText(text);
  }

private slots:
  void on_modelDataChanged()
  {
    updateText();
    emit globalCheckStateChanged(globalCheckState());
  }

  void on_itemPressed(const QModelIndex& index)
  {
    QStandardItem* item = m_model->itemFromIndex(index);

    if (item->checkState() == Qt::Checked) {
      item->setCheckState(Qt::Unchecked);
    } else {
      item->setCheckState(Qt::Checked);
    }
  }

public:
  class QCheckListStyledItemDelegate : public QStyledItemDelegate
  {
  public:
    QCheckListStyledItemDelegate(QObject* parent = 0)
      : QStyledItemDelegate(parent)
    {
    }

    void paint(QPainter* painter_, const QStyleOptionViewItem& option_, const QModelIndex& index_) const
    {
      QStyleOptionViewItem& refToNonConstOption = const_cast<QStyleOptionViewItem&>(option_);
      refToNonConstOption.showDecorationSelected = false;
      QStyledItemDelegate::paint(painter_, refToNonConstOption, index_);
    }
  };
};
