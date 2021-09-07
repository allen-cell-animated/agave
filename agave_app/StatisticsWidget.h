#pragma once

#include "renderlib/Status.h"

#include <QGridLayout>
#include <QtWidgets/QTreeWidget>

class CStatus;

class QStatisticsWidget : public QTreeWidget
{
  Q_OBJECT

public:
  QStatisticsWidget(QWidget* pParent = NULL);

  QSize sizeHint() const;

  void Init(void);
  void ExpandAll(const bool& Expand);

  void set(std::shared_ptr<CStatus> status);

private:
  void PopulateTree(void);
  QTreeWidgetItem* AddItem(QTreeWidgetItem* pParent,
                           const QString& Property,
                           const QString& Value = "",
                           const QString& Unit = "",
                           const QString& Icon = "");
  void UpdateStatistic(const QString& Group,
                       const QString& Name,
                       const QString& Value,
                       const QString& Unit,
                       const QString& Icon = "");
  QTreeWidgetItem* FindItem(const QString& Name);
  void RemoveChildren(const QString& Name);

public:
  void OnRenderBegin(void);
  void OnRenderEnd(void);
  void OnStatisticChanged(const QString& Group,
                          const QString& Name,
                          const QString& Value,
                          const QString& Unit,
                          const QString& Icon = "");

private:
  QGridLayout m_MainLayout;

  class MyStatusObserver : public IStatusObserver
  {
  public:
    virtual ~MyStatusObserver() {}
    virtual void RenderBegin(void) { mWidget->OnRenderBegin(); }
    virtual void RenderEnd(void) { mWidget->OnRenderEnd(); }
    virtual void PreRenderFrame(void) {}
    virtual void PostRenderFrame(void) {}
    virtual void RenderPause(const bool& Paused) {}
    virtual void Resize(void) {}
    virtual void LoadPreset(const std::string& PresetName) {}
    virtual void StatisticChanged(const std::string& Group,
                                  const std::string& Name,
                                  const std::string& Value,
                                  const std::string& Unit = "",
                                  const std::string& Icon = "")
    {
      mWidget->OnStatisticChanged(QString::fromStdString(Group),
                                  QString::fromStdString(Name),
                                  QString::fromStdString(Value),
                                  QString::fromStdString(Unit),
                                  QString::fromStdString(Icon));
    }
    // convoluted, yes
    QStatisticsWidget* mWidget;
  } mStatusObserver;

  std::shared_ptr<CStatus> mStatusObject;
};
