<template>
  <div class="about">
    <div class="display" v-if="$route.params.sequence">
      <div class="display_sequence">
        <div class="title">
          <h3 style="margin:5px; font-weight:700">Antibody's sequence</h3> 
          <h3 style="margin:auto; margin-left:10px"> Length: {{ $route.params.sequence.length }} aa</h3>
        </div>
        <div class="sequence">
          {{ $route.params.sequence }}
        </div>
      </div>
      <div class="display_structure">
        <div class="title">
          <h3 style="margin:5px; font-weight:700">3D structure</h3> 
          <h3 style="margin:auto; margin-left:10px"> From PDB closest sequence {{ id }}</h3>
        </div>
        
      </div>
    </div>
    <div v-else class="display_retrieve">
      <InputView
        text="Enter an antigene's peptidic sequence"
        title="Antibody input" 
        @updateText="updateText"
        @error="setInputError"
      />
      <div class="action">
        <Action title="Retrieve antibody data" :activated="antibody.length > 0 &!getInputError" @action="gotodisplay"/>
      </div>
    </div>
  </div>
</template>
<script>
import Action from '../components/Action.vue'
import InputView from '../components/Input.vue'
export default {
  name:'Display',
  components: {
    InputView,
    Action
  },

  data() {
    return {
      antibody: '',
      id: 'Nothing for now',
      isLoaded: false,
    }
  },
  computed: {
    getIsLoaded() {
      return this.isLoaded
    }
  },
  methods: {
    updateText(text) {
      this.error = false
      this.antibody = text
    },
    setInputError() {
      this.error = true
    },
    gotodisplay(event) {
      this.$router.push({ path: '/display/' + this.antibody})
    },
  }
}
</script>

<style>
.display_sequence {
  display: block;
  align-items: center;
  background-color: #38a3a5;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px;
  color: white
}
.display_retrieve {
  display: block;
  align-items: center;
  border: 1px solid lightgrey;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px
}
.sequence {
  background-color: white;
  height: 10rem;
  overflow-wrap: break-word;
  overflow: auto;
  border-radius: 5px;
  color: #3f5060;
  padding: 5px
}
.title {
  display:flex;
  margin-bottom:10px;
  justify-content: center;
}
.display_structure {
  display: block;
  align-items: center;
  background-color: #38a3a5;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px;
  color: white
}
</style>
