<template>
  <div class="generation">
    <div class="input">
      <InputVue 
        :text="inputText"
        title="Antigene input"
        @updateText="updateText"
        @error="setInputError"
      />
    </div>
    <div class="action">
      <Action title="Launch generation" :activated="Boolean(isValidText&!getInputError)" @action="sendRequest"/>
    </div>
    <div class="table-view" v-if="hasDatas">
        <h3 style="margin-bottom:5px">
          Generated Antibodies
        </h3>

      <TableView :datas="getTableDatas"/>
    </div>
    <div v-else-if="getIsLoading">
      <div class="center">
        <img  src="@/assets/loading.gif" alt="Loading" />
      </div>
      <div class="center">
        <h3>Generating data ...</h3>
      </div>
    </div>
    <div v-if="getIsError">
      An error occured: {{ error }}
    </div>
    <!--
    <div class="export">
      <vue-json-to-csv
        :json-data="getTableJson"
        csv-title="generated_antibodies"
        @success="val => console.log(val)"
        @error="val => console.error(val)"
      >
          <button class="export_button">
            Export 
          </button>
      </vue-json-to-csv>
    </div>
    -->
  </div>
</template>

<script>
import InputVue from "../components/Input.vue"
import Action from "../components/Action.vue"
import TableView from "../components/Table.vue"
import VueJsonToCsv from 'vue-json-to-csv'
import api from '@/api/api.js'
export default {
    name: 'GenerateView',
    data() {
      return {
        antigene: '',
        isLoading: false,
        isError:false,
        error:'',
        tableDatas: [],
        tableJson: {},
        inputError: false,
        inputText: "Enter an antigene's peptidic sequence"
      }
    },
    components: {
      InputVue,
      Action,
      TableView,
      VueJsonToCsv,
    },
    computed: {
      isValidText: function() {
        return Boolean(this.antigene.length > 0)
      },
      getTableJson: function() {
        return this.tableJson
      },
      getIsLoading: function() {
        return this.isLoading
      },
      getIsError: function() {
        return this.isError
      },
      hasDatas: function() {
        return this.tableDatas.length > 0
      },
      getTableDatas: function() {
        return this.tableDatas
      },
      getInputError: function() {
        return this.inputError
      }
    },
    mounted() {
      if (this.$store.state.antigenesTable.length > 0) {
        this.tableDatas = this.$store.state.antigenesTable
      }
      if (this.$store.state.sequence.length > 0) {
        this.inputText = this.$store.state.sequence
      }
      if (this.$store.state.json.length > 0) {
        this.dataJson = this.$store.state.json
      }
    },
    methods: {
      updateText(emited) {
        this.antigene = emited
        this.inputError = false
      },
      setInputError() {
        this.inputError = true
      },
      transpose(table) {
        var new_table = []
        const num_results = table[0].length
        for (var i = 0; i < num_results; i++) {
          var col_values =  table.map(x => x[i])
          new_table.push(col_values)
        }
        return new_table
      },
      parse_result(result) {
        var table = []
        var keys = Object.keys(result)
        keys.forEach(function(key){
          const current_col = result[key]
          var temp = []
          for (const col in Object.keys(current_col)) {
            temp.push(current_col[col])
          }
          table.push(temp)
        });
        return this.transpose(table)
      },
      toJson(datas) {
        const json = []
        for (var line in datas) {
          json.push({'Sequence':line[0], 'Score':line[1]})
        }
        return JSON.stringify(json)
      },
      async sendRequest () {
        try {
          this.isError = false
          this.isLoading = true
          const res = await api.generate.getGenerateData(this.antigene)
          this.isLoading = false
          this.tableDatas = this.parse_result(res)
          this.tableJson = this.toJson(this.tableDatas)
          this.$store.commit('setAntigenesTable', {data:this.tableDatas})
          this.$store.commit('setSequence', {data:this.antigene})
          this.$store.commit('setJson', {data:this.tableJson})
        } catch (error) {
          this.isLoading = false
          this.error = error
          this.isError = true
        }
      },
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

.loader {
  margin:auto;
  display: block;
  justify-content: center;
}

.export {
  margin:auto;
  width: 100%;
  display: flex;
  justify-content: center;
}

.export_button {
  background-color:#409d9e;
  border: none;
  border-radius: 2px;
  color: white;
  padding:5px;
  font-weight: 700;
}
.center {
  margin: auto;
  display: flex;
  justify-content: center;
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