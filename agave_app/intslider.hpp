/**
 * MIT License
 *
 * Copyright (c) 2023 Niels Bugel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include "valueslider.hpp"

namespace ValueSliders {

    class IntSlider : public ValueSlider<int> {
    Q_OBJECT
    public:
        IntSlider(QString name, int value);

        IntSlider(QString name, int value, int min, int max, BoundMode boundMode = BoundMode::UPPER_LOWER);

        [[nodiscard]] int transform(int val) const override;

        int convertString(const QString &string, bool &ok) override;

        [[nodiscard]] QString createString(int val) const override;

        void emitValueUpdated(int val) override;

        [[nodiscard]] int getValueByPosition(int x) override;

        void mousePressEvent(QMouseEvent *event) override;

    Q_SIGNALS:

        void valueUpdated(int value);

    private:
        double moveValue_;

        void updateBounds();
    };
} // ValueSliders
