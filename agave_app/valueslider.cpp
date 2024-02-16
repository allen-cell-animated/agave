#include "valueslider.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QTimer>
#include <QStyleOptionProgressBar>
#include <QElapsedTimer>
#include <utility>
#include <QApplication>

template<class T>
ValueSliders::ValueSlider<T>::ValueSlider(QString name, T value)
        : name_(std::move(name)),
          value_(value) {
    if (value > 0) {
        min_ = 0;
        max_ = value_ * 2;
    } else if (value < 0) {
        min_ = value_ * 2;
        max_ = 0;
    } else {
        min_ = -1;
        max_ = 1;
    }
    init();
}

template<class T>
ValueSliders::ValueSlider<T>::ValueSlider(QString name, T value, T min, T max, BoundMode boundMode)
        : boundMode_(boundMode),
          name_(std::move(name)),
          value_(value),
          min_(min),
          max_(max) {

    if (min > max) {
        throw std::invalid_argument(
                QString("ValueSlider min val cannot be greater than max val.\nMin: %1\nMax: %2\n").arg(min,
                                                                                                       max).toStdString());
    }
    init();
}


template<class T>
ValueSliders::ValueSlider<T>::~ValueSlider() {
    if (slidingHover_) {
        QApplication::restoreOverrideCursor();
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::init() {
    setFocusPolicy(Qt::StrongFocus);
    blinkerTimer_ = std::make_shared<QTimer>(this);
    connect(blinkerTimer_.get(), &QTimer::timeout, this, &ValueSliders::ValueSlider<T>::toggleBlinkerVisibility);
}

template<class T>
void ValueSliders::ValueSlider<T>::toggleBlinkerVisibility() {
    blinkerVisible_ = !blinkerVisible_;
    update();
}

template<class T>
QString ValueSliders::ValueSlider<T>::text() const {
    return "";
}

template<class T>
void ValueSliders::ValueSlider<T>::startTyping() {
    setFocus();
    QApplication::setOverrideCursor(Qt::IBeamCursor);
    grabKeyboard();
    setEnabled(true);
    typeInput_ = "";
    typing_ = true;
    blinkerTimer_->start(blinkerInterval_);
    update();
}

template<class T>
void ValueSliders::ValueSlider<T>::stopTyping() {
    releaseKeyboard();
    QApplication::restoreOverrideCursor();
    blinkerTimer_->stop();
    typing_ = false;
    setVal(value_);
    update();
}

template<class T>
void ValueSliders::ValueSlider<T>::paintEvent(QPaintEvent *event) {
    QProgressBar::paintEvent(event);

    QPainter painter(this);
    QStyleOptionProgressBar option;
    initStyleOption(&option);
    if (typing_) {
        QRect rect = style()->subElementRect(QStyle::SE_ProgressBarContents, &option, this);
        rect.setX(rect.x() + padding_);

        bool start = typeInput_ == "";
        QFontMetrics fontMetrics(font());

        painter.save();
        QString nameText = start ? QString::number(value_) : typeInput_;
        QSize textSize = fontMetrics.size(Qt::TextSingleLine, nameText);

        QRect textRect = rect;
        textRect.setWidth(textSize.width() + padding_);
        textRect.moveLeft(padding_ / 2);
        QColor highlightColor = palette().color(QPalette::Highlight);

        painter.setBrush(QBrush(highlightColor));
        painter.setPen(Qt::NoPen);
        if (start) {
            painter.drawRect(textRect);
        }

        if (blinkerVisible_) {
            QRect blinkerRect(textRect.x() + textRect.width() - padding_ / 2, textRect.y(), 2, textRect.height());
            painter.fillRect(blinkerRect, palette().color(QPalette::WindowText));
        }

        painter.restore();

        painter.drawText(rect, Qt::AlignLeft | Qt::AlignVCenter, nameText);
    } else {
        QRect rect = style()->subElementRect(QStyle::SE_ProgressBarContents, &option, this);
        rect.setX(rect.x() + padding_);
        QString nameText = name_;
        painter.drawText(rect, Qt::AlignLeft | Qt::AlignVCenter, nameText);

        QString valueText = createString(value_);
        QRect valueRect = rect.adjusted(QFontMetrics(font()).horizontalAdvance(nameText), 0, -padding_, 0);
        painter.drawText(valueRect, Qt::AlignRight | Qt::AlignVCenter, valueText);
    }

}

template<class T>
void ValueSliders::ValueSlider<T>::mousePressEvent(QMouseEvent *event) {
    setFocus();
    if (typing_) {
        return;
    }
    if (event->button() == Qt::LeftButton) {
        QApplication::setOverrideCursor(Qt::BlankCursor);
        startPos_ = QCursor::pos();
        oldPos_ = event->pos().x();
        mouseMoved_ = false;
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::mouseMoveEvent(QMouseEvent *event) {
    if (typing_) {
        return;
    }
    if (event->buttons() & Qt::LeftButton) {
        int diff = event->pos().x() - oldPos_;
        QCursor::setPos(startPos_);
        updateValueByPosition(diff);
        mouseMoved_ = true;
        return;
    }
}

template<class T>
int ValueSliders::ValueSlider<T>::getXPosByVal() const {
    double val = std::clamp(transform(value_), minimum(), maximum());
    double ratio = val / (maximum() - minimum());
    int xOffset = std::clamp(int(ratio * width()), padding_, width() - padding_);
    return mapToGlobal(pos()).x() + xOffset;
}

template<class T>
void ValueSliders::ValueSlider<T>::mouseReleaseEvent(QMouseEvent *event) {
    if (mouseMoved_) {
        if (event->button() == Qt::LeftButton) {
            QCursor::setPos(startPos_);
            updateValueByPosition(event->pos().x() - oldPos_);
        }
        QApplication::restoreOverrideCursor();
    } else {
        QApplication::restoreOverrideCursor();
        if (!typing_) {
            startTyping();
        }
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::updateValueByPosition(int x) {
    if (x == 0) {
        return;
    }
    setVal(getValueByPosition(x));
    setEnabled(true);
}

template<class T>
void ValueSliders::ValueSlider<T>::mouseDoubleClickEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton && !typing_) {
        startTyping();
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::keyPressEvent(QKeyEvent *event) {
    if (typing_) {
        event->accept();
        if (event->key() == Qt::Key_Escape) {
            stopTyping();
            setEnabled(true);
            return;
        }
        if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
            submitTypedInput();
            return;
        }
        if (event->key() == Qt::Key_Backspace) {
            typeInput_.chop(1);
            update();
            return;
        }
        typeInput_ += event->text();
        update();
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::submitTypedInput() {
    bool ok;
    T newVal = convertString(typeInput_, ok);
    if (ok) {
        setVal(newVal);
    }
    stopTyping();
    setEnabled(true);
    QApplication::restoreOverrideCursor();
    if (underMouse() && !slidingHover_) {
        QApplication::setOverrideCursor(Qt::SizeHorCursor);
        slidingHover_ = true;
    } else {
        slidingHover_ = false;
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::focusOutEvent(QFocusEvent *event) {
    if (typing_) {
        submitTypedInput();
    }
    QApplication::restoreOverrideCursor();
    slidingHover_ = false;
}

template<class T>
T ValueSliders::ValueSlider<T>::boundVal(T value) const {
    switch (boundMode_) {
        case BoundMode::UNCHECKED:
            return value;
        case BoundMode::LOWER_ONLY:
            return std::max(value, min_);
        case BoundMode::UPPER_ONLY:
            return std::min(value, max_);
        case BoundMode::UPPER_LOWER:
            return std::clamp(value, min_, max_);
        default:
            return value;
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::setVal(T value) {
    if (value_ == value) {
        return;
    }
    value_ = boundVal(value);
    if (boundMode_ == BoundMode::UPPER_LOWER) {
        setValue(std::clamp(transform(value_), minimum(), maximum()));
    }
    emitValueUpdated(value_);
    update();
}

template<class T>
T ValueSliders::ValueSlider<T>::getVal() const {
    return value_;
}

template<class T>
void ValueSliders::ValueSlider<T>::enterEvent(QEnterEvent *event) {
    if (!typing_ && !slidingHover_) {
        QApplication::setOverrideCursor(Qt::SizeHorCursor);
        slidingHover_ = true;
    }
}

template<class T>
void ValueSliders::ValueSlider<T>::leaveEvent(QEvent *event) {
    if (!typing_) {
        QApplication::restoreOverrideCursor();
        slidingHover_ = false;
    }
}

template
class ValueSliders::ValueSlider<int>;

template
class ValueSliders::ValueSlider<double>;
