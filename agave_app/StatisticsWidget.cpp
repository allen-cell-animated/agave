#include "StatisticsWidget.h"

#include "Status.h"

#include <QHeaderView>

QStatisticsWidget::QStatisticsWidget(QWidget* pParent)
  : QTreeWidget(pParent)
  , m_MainLayout()
  , mStatusObject(nullptr)
{
  mStatusObserver.mWidget = this;

  // Set the size policy, making sure the widget fits nicely in the layout
  setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

  // Status and tooltip
  setToolTip(tr("Statistics"));
  setStatusTip(tr("Statistics"));

  // Configure tree
  setColumnCount(2);

  QStringList ColumnNames;
  ColumnNames << "Property"
              << "Value";
  setHeaderLabels(ColumnNames);

  // Configure headers
  //	header()->setResizeMode(0, QHeaderView::ResizeToContents);
  //	header()->setResizeMode(1, QHeaderView::ResizeToContents);
  //	header()->setResizeMode(2, QHeaderView::ResizeToContents);
  header()->resizeSection(0, 160);
  header()->resizeSection(1, 250);
  // header()->setWindowIcon(GetIcon("table-export"));
  header()->setVisible(false);

  PopulateTree();

  // Notify us when rendering begins and ends, and before/after each rendered frame
  // connect(&gStatus, SIGNAL(StatisticChanged(const QString&, const QString&, const QString&, const QString&, const
  // QString&)), this, SLOT(OnStatisticChanged(const QString&, const QString&, const QString&, const QString&, const
  // QString&)));
}

void
QStatisticsWidget::set(std::shared_ptr<CStatus> status)
{
  if (status != mStatusObject && mStatusObject != nullptr) {
    mStatusObject->removeObserver(&mStatusObserver);
  }
  mStatusObject = status;
  if (mStatusObject) {
    mStatusObject->addObserver(&mStatusObserver);
  }
}

QSize
QStatisticsWidget::sizeHint() const
{
  return QSize(250, 900);
}

void
QStatisticsWidget::PopulateTree(void)
{
  // Populate tree with top-level items
  AddItem(NULL, "Performance", "", "", "application-monitor");
  AddItem(NULL, "Volume", "", "", "grid");
}

QTreeWidgetItem*
QStatisticsWidget::AddItem(QTreeWidgetItem* pParent,
                           const QString& Property,
                           const QString& Value,
                           const QString& Unit,
                           const QString& Icon)
{
  // Create new item
  QTreeWidgetItem* pItem = new QTreeWidgetItem(pParent);

  // Set item properties
  pItem->setText(0, Property);
  pItem->setText(1, Value + " " + Unit);
  // pItem->setIcon(0, GetIcon(Icon));

  if (!pParent)
    addTopLevelItem(pItem);

  return pItem;
}

void
QStatisticsWidget::UpdateStatistic(const QString& Group,
                                   const QString& Name,
                                   const QString& Value,
                                   const QString& Unit,
                                   const QString& Icon)
{
  QTreeWidgetItem* pGroup = FindItem(Group);

  if (!pGroup) {
    pGroup = AddItem(NULL, Group);

    AddItem(pGroup, Name, Value, Unit, Icon);
  } else {
    bool Found = false;

    for (int i = 0; i < pGroup->childCount(); i++) {
      if (pGroup->child(i)->text(0) == Name) {
        pGroup->child(i)->setText(1, Value + " " + Unit);
        Found = true;
      }
    }

    if (!Found)
      AddItem(pGroup, Name, Value, Unit, Icon);
  }
}

void
QStatisticsWidget::OnRenderBegin(void)
{
  // Expand all tree items
  ExpandAll(true);
}

void
QStatisticsWidget::OnRenderEnd(void)
{
  // Collapse all tree items
  ExpandAll(false);

  // Remove 2nd order children
  RemoveChildren("Performance");
  RemoveChildren("Volume");
  RemoveChildren("Memory");
  RemoveChildren("Camera");
  RemoveChildren("Graphics Card");
}

void
QStatisticsWidget::OnStatisticChanged(const QString& Group,
                                      const QString& Name,
                                      const QString& Value,
                                      const QString& Unit /*= ""*/,
                                      const QString& Icon /*= ""*/)
{
  UpdateStatistic(Group, Name, Value, Unit, Icon);
}

void
QStatisticsWidget::ExpandAll(const bool& Expand)
{
  QList<QTreeWidgetItem*> Items = findItems("*", Qt::MatchRecursive | Qt::MatchWildcard, 0);

  foreach (QTreeWidgetItem* pItem, Items)
    pItem->setExpanded(Expand);
}

QTreeWidgetItem*
QStatisticsWidget::FindItem(const QString& Name)
{
  QList<QTreeWidgetItem*> Items = findItems(Name, Qt::MatchRecursive, 0);

  if (Items.size() <= 0)
    return NULL;
  else
    return Items[0];
}

void
QStatisticsWidget::RemoveChildren(const QString& Name)
{
  QTreeWidgetItem* pItem = FindItem(Name);

  if (pItem)
    qDeleteAll(pItem->takeChildren());
}
