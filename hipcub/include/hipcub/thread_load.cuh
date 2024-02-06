#ifndef THREADLOAD
#define THREADLOAD

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval; // We removed the intialization to 0 because some of the data types cannot be initialized to 0.
    // Also, retval is set to ptr with the builtin
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename InputIteratorT>
HIPCUB_DEVICE __forceinline__
typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    using T  = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = ThreadLoad<MODIFIER>(&(*itr));
    return retval;
}

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T
ThreadLoad(T * ptr)
{
    return AsmThreadLoad<MODIFIER, T>(ptr);
}

#endif