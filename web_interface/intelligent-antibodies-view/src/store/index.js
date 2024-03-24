import { createStore } from 'vuex'

export default createStore({
  state: {
    antigenesTable: [],
    sequence: '',
    json: []
  },
  mutations: {
    setAntigenesTable(state, payload) {
      state.antigenesTable = payload.data
    },
    setSequence(state, payload) {
      state.sequence= payload.data
    },
    setJson(state, payload) {
      state.json= payload.data
    },
  }
})
