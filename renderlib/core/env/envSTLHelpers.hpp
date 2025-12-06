#pragma once

#include <algorithm> // for_each
#include <functional>

//============================================================================
//	envSTLHelpers
//============================================================================
namespace envSTLHelpers {

//----------------------------------------------------------------------------
//	DeleteContainer is meant to be used on containers of pointers.  It will
//	delete every object in the container and empty the container.
//----------------------------------------------------------------------------
template<class C>
void
DeleteContainer(C& io_Container)
{
  typename C::iterator it = io_Container.begin();
  typename C::iterator end = io_Container.end();

  while (it != end) {
    delete *it;
    ++it;
  }

  io_Container.clear();
}

//----------------------------------------------------------------------------
//	copy_if is the "missing" STL function which copies only if the predicate
//	returns true.
//----------------------------------------------------------------------------
template<class In, class Out, class Pred>
Out
copy_if(In first, In last, Out res, Pred p)
{
  while (first != last) {
    if (p(*first))
      *res++ = *first;
    ++first;
  }

  return res;
}

//----------------------------------------------------------------------------
//	ForEachPair calls the given function object for each unique pair
//	combination of objects in the given sequence.  (Objects are not paired
//	with themselves).  This can be visualized as all of the combinations
//	in the lower triangle of a matrix (not including the diagonal).
//----------------------------------------------------------------------------
template<class Iterator, class Function>
inline void
ForEachPair(Iterator i_Begin, Iterator i_End, Function i_Function)
{
  if (i_Begin == i_End)
    return; //	no objects

  Iterator outer = i_Begin;
  Iterator next_to_last = i_End;
  --next_to_last;

  for (; outer != next_to_last; ++outer) {
    Iterator inner = outer;
    ++inner;

    for (; inner != i_End; ++inner)
      i_Function(*outer, *inner);
  }
}

//----------------------------------------------------------------------------
//	ForAll works the same way as if you called std::for_each passing
//  in the begin and end iterators.
//----------------------------------------------------------------------------
template<class Container, class Function>
inline void
ForAll(Container& i_Container, Function i_Function)
{
  std::for_each(i_Container.begin(), i_Container.end(), i_Function);
}

//----------------------------------------------------------------------------
//	ForAll works the same way as if you called std::for_each passing
//  in the begin and end iterators.
//----------------------------------------------------------------------------
template<class Container, class Function>
inline void
RForAll(Container& i_Container, Function i_Function)
{
  std::for_each(i_Container.rbegin(), i_Container.rend(), i_Function);
}

//----------------------------------------------------------------------------
//	ForAllForAll is used for nested containers.  It will call the function on
//	each item in the nested vectors.
//----------------------------------------------------------------------------
template<class Container, class Function>
inline void
ForAllForAll(Container& i_Container, Function i_Function)
{
  typename Container::iterator it, end = i_Container.end();
  for (it = i_Container.begin(); it != end; ++it) {
    ForAll(*it, i_Function);
  }
}

//----------------------------------------------------------------------------
//	RemoveOneValue
//----------------------------------------------------------------------------
template<class C, class V>
bool
RemoveOneValue(C& io_Container, const V& i_Value)
{
  typename C::iterator it = io_Container.begin();
  typename C::iterator end = io_Container.end();

  while (it != end) {
    if (*it == i_Value) {
      io_Container.erase(it);
      return true;
    }
    ++it;
  }
  return false;
}

//----------------------------------------------------------------------------
//	RemoveAllValues
//----------------------------------------------------------------------------
template<class C, class V>
void
RemoveAllValues(C& io_Container, const V& i_Value)
{
  typename C::iterator it = io_Container.begin();

  while (it != io_Container.end()) {
    if (*it == i_Value)
      it = io_Container.erase(it);
    else
      ++it;
  }
}

//----------------------------------------------------------------------------
//	DeleteOneValue
//----------------------------------------------------------------------------
template<class C, class V>
bool
DeleteOneValue(C& io_Container, const V& i_Value)
{
  typename C::iterator it = io_Container.begin();
  typename C::iterator end = io_Container.end();

  while (it != end) {
    if (*it == i_Value) {
      delete (*it);
      io_Container.erase(it);
      return true;
    }
    ++it;
  }
  return false;
}

//----------------------------------------------------------------------------
//	Contains - returns true if value exists in container
//----------------------------------------------------------------------------
template<class C, class V>
bool
Contains(C& io_Container, const V& i_Value)
{
  typename C::const_iterator it, end = io_Container.end();
  for (it = io_Container.begin(); it != end; ++it) {
    if (*it == i_Value) {
      return true;
    }
  }
  return false;
}

}
