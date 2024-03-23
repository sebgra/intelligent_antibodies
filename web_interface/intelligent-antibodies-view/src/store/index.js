import { createStore } from 'vuex'

export default createStore({
  state: {
    antigenesTable: [],
    sequence: ''
  },
  mutations: {
    setAntigenesTable(state, payload) {
      state.antigenesTable = payload.data
    },
    setSequence(state, payload) {
      state.sequence= payload.data
    },
  }
})
