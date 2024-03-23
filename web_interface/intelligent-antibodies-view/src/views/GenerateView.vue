<template>
  <div class="generation">
    <div class="input">
      <InputVue 
        text="Enter an antigene's peptidic sequence"
        @updateText="updateText"
      />
    </div>
    <div class="action">
      <Action :activated="isValidText"/>
    </div>
    <div class="table-view" v-if="hasDatas">
        <h3 style="margin-bottom:5px">
          Generated Antibodies
        </h3>

      <TableView :datas="getTableDatas"/>
    </div>
    <div v-else-if="getIsLoading">
      Loading
    </div>
  </div>
</template>

<script>
import InputVue from "../components/Input.vue"
import Action from "../components/Action.vue"
import TableView from "../components/Table.vue"
export default {
    name: 'GenerateView',
    data() {
      return {
        antigene: '',
        isLoading: false,
        tableDatas: [
          ['seq1', 'score1'],
          ['seq2', 'score2'],
          ['seq1', 'score1'],
          ['seq2', 'score2'],
          ['seq1', 'score1'],
          ['seq2', 'score2'],
          ['seq1', 'score1'],
          ['seq2', 'score2']
        ]
      }
    },
    components: {
      InputVue,
      Action,
      TableView
    },
    computed: {
      isValidText: function() {
        return this.antigene.length > 0
      },
      getIsLoading: function() {
        return this.isLoading
      },
      hasDatas: function() {
        return this.tableDatas.length > 0
      },
      getTableDatas: function() {
        return this.tableDatas
      }
    },
    methods: {
      updateText(emited) {
        this.antigene = emited
      }
    }
  }

</script>

<style>
.generation {
  display: block;
  align-items: center;
  border: 1px solid lightgrey;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px
}

.input {
  width: 100%;
}

.action {
  width: 100%;
  margin-bottom:10px;
}

.table-view {
  width: 100%;
  padding: 10px;
}
</style>
