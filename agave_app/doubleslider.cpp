#include "doubleslider.hpp"

#include <utility>
#include <QDebug>

ValueSliders::DoubleSlider::DoubleSlider(QString name, double value) : ValueSlider(std::move(name), value) {
    updateBounds();
}

ValueSliders::DoubleSlider::DoubleSlider(QString name, double value, double min, double max,
                                         ValueSliders::BoundMode boundMode) : ValueSlider(std::move(name), value, min,
                                                                                          max,
                                                                                          boundMode) {
    updateBounds();
}

void ValueSliders::DoubleSlider::updateBounds() {
    setMinimum(int(std::round(min_ * 100.0)));
    setMaximum(int(std::round(max_ * 100.0)));
    if(boundMode_ == BoundMode::UPPER_LOWER) {
        setValue(int(std::round(value_ * 100.0)));
    } else {
        setValue(minimum());
    }
}

int ValueSliders::DoubleSlider::transform(double val) const {
    return int(std::round(val * 100.0));
}

double ValueSliders::DoubleSlider::convertString(const QString &string, bool &ok) {
    return string.toDouble(&ok);
}

QString ValueSliders::DoubleSlider::createString(double val) const {
    return QString::number(val, 'f', 3);
}

void ValueSliders::DoubleSlider::emitValueUpdated(double val) {
    emit valueUpdated(val);
}

double ValueSliders::DoubleSlider::getValueByPosition(int x) {
    double ratio = static_cast<double>(x) / width();
    double val = ratio * (maximum() - minimum());
    return value_ + val / 100.0;
}
