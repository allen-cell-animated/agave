#pragma once

#include "CacheSettings.h"

#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QWidget>

class CacheSettingsWidget : public QWidget
{
  Q_OBJECT

public:
  explicit CacheSettingsWidget(QWidget* parent = nullptr);

  void setSettings(const CacheSettingsData& data);
  CacheSettingsData getSettings() const;

  QPushButton* applyButton() const { return m_applyButton; }
  QPushButton* clearDiskButton() const { return m_clearDiskButton; }

private slots:
  void browseForCacheDir();

private:
  QCheckBox* m_enableCache = nullptr;
  QCheckBox* m_enableDisk = nullptr;
  QSpinBox* m_ramLimitMB = nullptr;
  QSpinBox* m_diskLimitGB = nullptr;
  QLineEdit* m_cacheDirEdit = nullptr;
  QPushButton* m_browseButton = nullptr;
  QPushButton* m_applyButton = nullptr;
  QPushButton* m_clearDiskButton = nullptr;
};
